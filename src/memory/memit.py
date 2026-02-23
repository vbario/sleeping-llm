"""MEMIT engine — Mass-Editing Memory in a Transformer.

Implements the MEMIT algorithm for MLX, providing fast factual memory injection
during the wake phase. Acts as the "hippocampus" — fast, fragile memory that
gets consolidated into stable LoRA weights during sleep.

Key classes:
  FactTriple — subject/relation/object representation of a fact
  EditLedger — persists edit metadata to disk
  MemitEdit  — in-memory record of an applied MEMIT edit
  MemitEngine — core MEMIT implementation
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reference texts for covariance estimation — diverse topics to capture
# the model's typical activation patterns across different contexts.
COVARIANCE_REFERENCE_TEXTS = [
    "The theory of general relativity, proposed by Albert Einstein in 1915, describes gravity as the warping of spacetime by mass and energy.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy from sunlight in the chloroplasts of plant cells.",
    "The French Revolution of 1789 overthrew the monarchy and established the First Republic, fundamentally transforming French society and politics.",
    "DNA stores genetic information in a double helix structure, with base pairs of adenine-thymine and guanine-cytosine connected by hydrogen bonds.",
    "The Silk Road was an ancient network of trade routes connecting China to the Mediterranean, facilitating exchange of goods, ideas, and cultures.",
    "Quantum mechanics describes the behavior of matter and energy at the atomic scale, where particles exhibit wave-particle duality.",
    "The Amazon rainforest covers much of South America and contains the greatest biodiversity of any ecosystem on Earth.",
    "Classical music evolved through the Baroque, Classical, and Romantic periods, with composers like Bach, Mozart, and Beethoven.",
    "The human brain contains approximately 86 billion neurons connected by trillions of synapses, enabling thought, memory, and consciousness.",
    "Plate tectonics explains how Earth's lithosphere is divided into plates that move, creating earthquakes, volcanoes, and mountain ranges.",
    "The Industrial Revolution began in Britain in the late 18th century, transforming manufacturing through steam power and mechanization.",
    "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides.",
    "Climate change is driven primarily by greenhouse gas emissions from burning fossil fuels, deforestation, and industrial processes.",
    "Shakespeare wrote approximately 37 plays including Hamlet, Macbeth, and Romeo and Juliet, profoundly influencing English literature.",
    "The electromagnetic spectrum ranges from radio waves through microwaves, infrared, visible light, ultraviolet, X-rays, to gamma rays.",
    "Ancient Egypt developed along the Nile River, building pyramids and developing hieroglyphic writing over thousands of years of civilization.",
    "Machine learning algorithms improve their performance on tasks through experience, without being explicitly programmed for each specific case.",
    "The Periodic Table organizes chemical elements by atomic number, revealing patterns in their properties and chemical behavior.",
    "Democracy originated in ancient Athens, where citizens directly participated in governance and decision-making for their city-state.",
    "Ocean currents distribute heat around the globe, with the Gulf Stream warming Western Europe and influencing weather patterns worldwide.",
    "The Renaissance began in Italy in the 14th century, reviving interest in classical learning, art, and humanism across Europe.",
    "Fibonacci numbers appear throughout nature, from the spiral of shells to the arrangement of leaves and the branching of trees.",
    "The invention of the printing press by Gutenberg around 1440 revolutionized the spread of knowledge and literacy across Europe.",
    "Volcanic eruptions occur when magma from Earth's mantle reaches the surface, releasing lava, ash, and gases into the atmosphere.",
    "Jazz music originated in New Orleans in the early 20th century, blending African American musical traditions with European harmonies.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second, serving as a fundamental constant in physics.",
    "Coral reefs support about 25 percent of all marine species despite covering less than one percent of the ocean floor worldwide.",
    "The Hubble Space Telescope has provided unprecedented views of distant galaxies, nebulae, and other astronomical phenomena since 1990.",
    "Antibiotics discovered in the 20th century transformed medicine by providing effective treatments against bacterial infections.",
    "The Great Wall of China stretches over 13,000 miles and was built over many centuries to protect against invasions from the north.",
]


@dataclass
class FactTriple:
    """A single fact represented as (subject, relation, object).

    Example: ("Vladimir", "lives in", "Portland")
    """
    subject: str
    relation: str
    object: str
    source_exchange: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_prompt(self) -> str:
        """Convert to a natural language prompt (subject + relation)."""
        return f"{self.subject} {self.relation}"

    def to_target(self) -> str:
        """The expected completion."""
        return f" {self.object}"

    def to_question(self) -> str:
        """Convert to a natural language question."""
        relation_to_question = {
            "is named": f"What is the user's name?",
            "lives in": f"Where does {self.subject} live?",
            "works as": f"What does {self.subject} do for work?",
            "is a": f"What is {self.subject}?",
            "likes": f"What does {self.subject} like?",
            "dislikes": f"What does {self.subject} dislike?",
            "has": f"What does {self.subject} have?",
            "uses": f"What does {self.subject} use?",
            "is aged": f"How old is {self.subject}?",
            "'s favorite": f"What is {self.subject}'s favorite?",
        }
        for key, question in relation_to_question.items():
            if key in self.relation:
                return question
        return f"What is the relationship between {self.subject} and {self.object}?"

    def to_answer(self) -> str:
        """Convert to a natural language answer."""
        return f"{self.subject} {self.relation} {self.object}."

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "source_exchange": self.source_exchange,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: dict) -> "FactTriple":
        return FactTriple(
            subject=d["subject"],
            relation=d["relation"],
            object=d["object"],
            source_exchange=d.get("source_exchange"),
            timestamp=d.get("timestamp", time.time()),
        )


@dataclass
class MemitEdit:
    """Record of a single MEMIT edit operation (may contain multiple facts).

    layer_deltas and key_vectors are kept in memory only — not serialized.
    They are needed for reverting edits and scaling.
    """
    edit_id: str
    facts: List[FactTriple]
    layer_deltas: Dict[int, object] = field(default_factory=dict)  # layer_idx -> mx.array
    layer_indices: List[int] = field(default_factory=list)
    key_vectors: Dict[int, object] = field(default_factory=dict)  # layer_idx -> mx.array
    timestamp: float = field(default_factory=time.time)
    consolidated: bool = False
    scale: float = 1.0  # fraction of delta applied (1.0=full, 0.1=residual trace)
    consolidation_stage: int = 0  # 0=active, 1=consolidating (LoRA+residual), 2=consolidated

    def to_ledger_dict(self) -> dict:
        """Serialize metadata only (no arrays) for the ledger."""
        return {
            "edit_id": self.edit_id,
            "facts": [f.to_dict() for f in self.facts],
            "layer_indices": self.layer_indices,
            "timestamp": self.timestamp,
            "consolidated": self.consolidated,
            "scale": self.scale,
            "consolidation_stage": self.consolidation_stage,
        }


class EditLedger:
    """Persists MEMIT edit metadata to disk.

    Tracks which facts have been injected via MEMIT, which are still active,
    and which have been consolidated into LoRA during sleep.
    """

    def __init__(self, ledger_path: str):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._deltas_dir = self.ledger_path.parent / "deltas"
        self._deltas_dir.mkdir(parents=True, exist_ok=True)
        self._edits: List[dict] = []
        self.load()

    def record_edit(self, edit: MemitEdit):
        """Record a new MEMIT edit."""
        self._edits.append(edit.to_ledger_dict())
        self.save()

    def get_active_edits(self) -> List[dict]:
        """Return all non-consolidated edits (stage 0 and 1).

        Backward compat: missing scale/consolidation_stage default to 1.0/0.
        """
        active = []
        for e in self._edits:
            if not e.get("consolidated", False):
                # Ensure backward-compat defaults
                e.setdefault("scale", 1.0)
                e.setdefault("consolidation_stage", 0)
                active.append(e)
        return active

    def get_edit_count(self) -> int:
        """Return count of active (non-consolidated) edits."""
        return len(self.get_active_edits())

    def get_active_fact_count(self) -> int:
        """Return count of individual active facts across all edits."""
        return sum(len(e["facts"]) for e in self.get_active_edits())

    def mark_consolidated(self, edit_ids: List[str]):
        """Mark edits as consolidated (absorbed into LoRA)."""
        id_set = set(edit_ids)
        for edit in self._edits:
            if edit["edit_id"] in id_set:
                edit["consolidated"] = True
        self.save()

    def update_scale(self, edit_id: str, new_scale: float, stage: int):
        """Update scale and consolidation_stage for an edit."""
        for edit in self._edits:
            if edit["edit_id"] == edit_id:
                edit["scale"] = new_scale
                edit["consolidation_stage"] = stage
                break
        self.save()

    def get_consolidating_edits(self) -> List[dict]:
        """Return edits with consolidation_stage == 1 (partially consolidated)."""
        return [e for e in self._edits
                if not e.get("consolidated", False)
                and e.get("consolidation_stage", 0) == 1]

    def get_stage_counts(self) -> Dict[int, int]:
        """Return count of active edits per consolidation stage."""
        counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}
        for e in self._edits:
            if not e.get("consolidated", False):
                stage = e.get("consolidation_stage", 0)
                counts[stage] = counts.get(stage, 0) + 1
        return counts

    def get_facts_for_training(self) -> List[FactTriple]:
        """Return all active facts as FactTriple objects for training data generation."""
        facts = []
        for edit in self.get_active_edits():
            for fact_dict in edit["facts"]:
                facts.append(FactTriple.from_dict(fact_dict))
        return facts

    def clear_consolidated(self):
        """Remove all consolidated edits from the ledger."""
        self._edits = [e for e in self._edits if not e.get("consolidated", False)]
        self.save()

    def clear_all(self):
        """Clear the entire ledger."""
        self._edits = []
        self.save()

    def save_deltas(self, edit_id: str, layer_deltas: dict):
        """Serialize weight delta tensors to disk.

        Args:
            edit_id: The edit identifier
            layer_deltas: Dict of layer_idx -> tensor (mx.array or torch.Tensor)
        """
        import numpy as np
        delta_path = self._deltas_dir / f"{edit_id}.npz"
        arrays = {}
        for layer_idx, delta in layer_deltas.items():
            if hasattr(delta, 'numpy'):
                # torch.Tensor
                arr = delta.cpu().float().numpy()
            else:
                # mx.array
                import mlx.core as mx
                arr = np.array(delta.astype(mx.float32))
            arrays[str(layer_idx)] = arr
        np.savez(delta_path, **arrays)

    def load_deltas(self, edit_id: str) -> Optional[Dict[int, object]]:
        """Deserialize weight delta tensors from disk.

        Returns:
            Dict of layer_idx -> tensor, or None if not found.
            Returns numpy arrays — caller should convert to framework tensor.
        """
        import numpy as np
        delta_path = self._deltas_dir / f"{edit_id}.npz"
        if not delta_path.exists():
            return None
        data = np.load(delta_path)
        return {int(k): data[k] for k in data.files}

    def save(self):
        """Write ledger to disk."""
        with open(self.ledger_path, "w") as f:
            json.dump(self._edits, f, indent=2)

    def load(self):
        """Load ledger from disk."""
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path) as f:
                    self._edits = json.load(f)
            except (json.JSONDecodeError, ValueError):
                self._edits = []
        else:
            self._edits = []


def _detect_backend(backend):
    """Detect whether backend is MLX or PyTorch and return tensor ops."""
    backend_type = type(backend).__name__
    if backend_type == "TorchBackend":
        return "torch"
    return "mlx"


def _tensor_ops(backend_type):
    """Return a namespace of tensor operations for the detected backend.

    Provides: make_input_ids, stack, eye, inv, eval, matmul, transpose
    """
    class Ops:
        pass
    ops = Ops()

    if backend_type == "torch":
        import torch
        ops.make_input_ids = lambda tokens: torch.tensor([tokens], dtype=torch.long)
        ops.stack = torch.stack
        ops.eye = lambda n: torch.eye(n)
        ops.transpose = lambda x: x.T
        ops.matmul = lambda a, b: a @ b
        ops.negate = lambda x: -x

        def _inv(x):
            return torch.linalg.inv(x)
        ops.inv = _inv

        def _eval(*args):
            pass  # no-op for PyTorch (eager execution)
        ops.eval = _eval

        def _to_device(t, backend):
            # Use first parameter's device (safe for single or multi-GPU)
            dev = next(backend.model.parameters()).device
            return t.to(dev)
        ops.to_device = _to_device
    else:
        import mlx.core as mx
        ops.make_input_ids = lambda tokens: mx.array([tokens])
        ops.stack = mx.stack
        ops.eye = mx.eye
        ops.transpose = lambda x: x.T
        ops.matmul = lambda a, b: a @ b
        ops.negate = lambda x: -x

        def _inv(x):
            return mx.linalg.inv(x)
        ops.inv = _inv

        def _eval(*args):
            mx.eval(*[a for a in args if a is not None])
        ops.eval = _eval

        def _to_device(t, backend):
            return t  # no-op for MLX
        ops.to_device = _to_device

    return ops


class MemitEngine:
    """Core MEMIT implementation — backend-agnostic (MLX and PyTorch).

    Injects batches of facts into transformer MLP layers by computing
    weight deltas distributed across multiple target layers.

    Algorithm (for batch of facts across target layers L_first...L_last):
    1. Compute key vectors at each target layer (MLP input at subject's last token)
    2. Compute target values at critical layer (L_last) — the hidden state
       that would cause the model to generate the desired object
    3. Distribute updates backwards from L_last to L_first:
       - At each layer, compute ΔW that absorbs a fraction of remaining residual
       - ΔW = R @ K^T @ (K @ K^T + λI)^{-1}
    4. Apply deltas to MLP down_proj weights
    5. Verify recall
    """

    def __init__(self, config, backend, ledger: EditLedger):
        self.config = config
        self.backend = backend
        self.ledger = ledger

        memit_config = config.get("memit", {}) or {}
        self.target_layers = memit_config.get("target_layers", [8, 9, 10, 11, 12, 13, 14, 15])
        self.lambda_reg = memit_config.get("lambda_reg", 0.1)
        self.max_active_edits = memit_config.get("max_active_edits", 50)
        self.enabled = memit_config.get("enabled", True)
        self.target_module = memit_config.get("target_module", "down_proj")
        self.covariance_samples = memit_config.get("covariance_samples", 0)
        # Scale factor for residuals to compensate for weight edit losses
        # (null-space constraints + regularization absorb ~50% of the signal)
        self.residual_scale = memit_config.get("residual_scale", 2.0)
        # v* optimization hyperparameters
        self.v_lr = memit_config.get("v_lr", 0.5)
        self.v_steps = memit_config.get("v_steps", 30)
        self.v_kl_factor = memit_config.get("v_kl_factor", 0.0625)

        # Detect backend and get tensor ops
        self._backend_type = _detect_backend(backend)
        self._ops = _tensor_ops(self._backend_type)

        # In-memory store of active edits (with delta arrays for revert)
        self._active_edits: List[MemitEdit] = []

        # Diagonal covariance estimates per layer (for regularization)
        self._cov_diagonal: Dict[int, object] = {}

        # Dequantize target layers for MEMIT (4-bit weights can't accept float deltas)
        if self.enabled and hasattr(backend, "dequantize_layer"):
            self._dequantize_target_layers()

        # Estimate covariance if configured
        if self.enabled and self.covariance_samples > 0:
            self._estimate_covariance_diagonal()

    def _dequantize_target_layers(self):
        """Dequantize target MLP layers so MEMIT can modify float weights directly."""
        num_layers = self.backend.get_num_layers()
        target_layers = [l for l in self.target_layers if l < num_layers]
        dequantized = 0
        for layer_idx in target_layers:
            if self.backend.dequantize_layer(layer_idx, self.target_module):
                dequantized += 1
        if dequantized > 0:
            print(f"  MEMIT: dequantized {dequantized} {self.target_module} layers for weight editing")

    def _estimate_covariance_diagonal(self):
        """Estimate per-dimension activation variance at each target layer.

        Uses reference texts to collect intermediate MLP activations (input to
        down_proj) and computes per-dimension variance. This captures which
        activation dimensions are heavily used by the model — those dimensions
        get stronger regularization to prevent MEMIT edits from corrupting
        the model's normal behavior.

        Results are cached to data/memit/cov_diag_layer_{n}.npy.

        The variance vector σ² is used in the Woodbury-based delta formula:
          ΔW = R^T @ S^{-1} @ K_w
        where K̃ = K/(√λ·σ), K_w = K/(λ·σ²), S = I + K̃@K̃^T
        """
        ops = self._ops
        cache_dir = Path(self.config.get("paths.memit_data", "data/memit"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        num_layers = self.backend.get_num_layers()
        target_layers = [l for l in self.target_layers if l < num_layers]

        # Check if all layers are cached
        all_cached = True
        for layer_idx in target_layers:
            cache_file = cache_dir / f"cov_diag_layer_{layer_idx}.json"
            if not cache_file.exists():
                all_cached = False
                break

        if all_cached:
            # Load from cache
            for layer_idx in target_layers:
                cache_file = cache_dir / f"cov_diag_layer_{layer_idx}.json"
                with open(cache_file) as f:
                    var_list = json.load(f)
                if self._backend_type == "mlx":
                    import mlx.core as mx
                    self._cov_diagonal[layer_idx] = mx.array(var_list)
                else:
                    import torch
                    self._cov_diagonal[layer_idx] = torch.tensor(var_list)
            print(f"  MEMIT: loaded cached covariance for {len(target_layers)} layers")
            return

        print(f"  MEMIT: estimating covariance from {len(COVARIANCE_REFERENCE_TEXTS)} reference texts...")

        # Collect activations per layer
        layer_activations = {l: [] for l in target_layers}

        for text_idx, text in enumerate(COVARIANCE_REFERENCE_TEXTS):
            tokens = self.backend.tokenizer.encode(text)
            input_ids = ops.to_device(ops.make_input_ids(tokens), self.backend)

            for layer_idx in target_layers:
                _, mlp_input, _ = self.backend.forward_to_layer(input_ids, layer_idx)
                intermediate = self.backend.compute_mlp_intermediate(mlp_input, layer_idx)
                # intermediate: [1, seq_len, intermediate_size]
                seq_len = intermediate.shape[1]
                for pos in range(seq_len):
                    act = intermediate[0, pos, :]
                    layer_activations[layer_idx].append(act)

            if (text_idx + 1) % 10 == 0:
                print(f"    processed {text_idx + 1}/{len(COVARIANCE_REFERENCE_TEXTS)} texts")

        # Compute per-dimension variance and cache
        for layer_idx in target_layers:
            acts = layer_activations[layer_idx]
            if not acts:
                continue

            acts_matrix = ops.stack(acts)  # [num_samples, intermediate_size]
            ops.eval(acts_matrix)

            num_samples = acts_matrix.shape[0]

            if self._backend_type == "mlx":
                import mlx.core as mx
                # Compute in float32 to avoid overflow (float16 max is ~65504,
                # squared activations can easily exceed that)
                acts_f32 = acts_matrix.astype(mx.float32)
                mx.eval(acts_f32)
                mean = mx.mean(acts_f32, axis=0)
                mx.eval(mean)
                centered = acts_f32 - mean
                variance = mx.mean(centered * centered, axis=0)
                mx.eval(variance)
                # Clamp minimum variance to prevent division by zero
                variance = mx.maximum(variance, mx.array(1e-6, dtype=mx.float32))
                # Keep as float32 for numerical stability in Woodbury formula
                mx.eval(variance)
            else:
                import torch
                variance = torch.var(acts_matrix.float(), dim=0)
                variance = torch.clamp(variance, min=1e-6)

            self._cov_diagonal[layer_idx] = variance

            # Cache to disk (as JSON list for portability)
            cache_file = cache_dir / f"cov_diag_layer_{layer_idx}.json"
            if self._backend_type == "mlx":
                var_list = variance.tolist()
            else:
                var_list = variance.cpu().tolist()
            with open(cache_file, "w") as f:
                json.dump(var_list, f)

            if self._backend_type == "mlx":
                v_min = mx.min(variance).item()
                v_max = mx.max(variance).item()
                v_mean = mx.mean(variance).item()
            else:
                v_min = variance.min().item()
                v_max = variance.max().item()
                v_mean = variance.mean().item()

            print(f"    Layer {layer_idx}: {num_samples} samples, "
                  f"var range [{v_min:.4f}, {v_max:.4f}], mean={v_mean:.4f}")

        print(f"  MEMIT: covariance estimated and cached for {len(target_layers)} layers")

    def inject_facts(self, facts: List[FactTriple]) -> Optional[MemitEdit]:
        """Batch injection — the primary MEMIT method.

        Processes all facts simultaneously across all target layers.

        Args:
            facts: List of FactTriple objects to inject

        Returns:
            MemitEdit record, or None if injection failed or MEMIT disabled
        """
        if not self.enabled or not facts:
            return None

        ops = self._ops

        # Clamp target layers to actual model size
        num_layers = self.backend.get_num_layers()
        target_layers = [l for l in self.target_layers if l < num_layers]
        if not target_layers:
            return None

        critical_layer = target_layers[-1]  # L_last

        # Step 1: Compute key vectors at each target layer (with null-space constraints)
        keys_per_layer = {}  # layer_idx -> (keys_matrix, target_mask)
        for layer_idx in target_layers:
            result = self._compute_keys(facts, layer_idx)
            if result is None:
                return None
            keys_per_layer[layer_idx] = result

        # Step 2: Compute target values at critical layer
        target_values = self._compute_target_values(facts, critical_layer)
        if target_values is None:
            return None

        # Step 3: Compute current values at critical layer
        current_values = self._compute_current_values(facts, critical_layer)
        if current_values is None:
            return None

        # Initial residual for target positions = target - current, scaled
        # Scale compensates for signal loss from null-space constraints + regularization
        # Ensure same device (multi-GPU: target/current may land on different devices)
        if self._backend_type != "mlx" and hasattr(target_values, 'device'):
            current_values = current_values.to(target_values.device)
        fact_residuals = self.residual_scale * (target_values - current_values)

        # Diagnostic: check residual magnitude
        if self._backend_type == "mlx":
            import mlx.core as mx
            r_norm = mx.sqrt(mx.sum(fact_residuals * fact_residuals)).item()
        else:
            import torch
            r_norm = torch.norm(fact_residuals).item()
        first_keys, first_mask = list(keys_per_layer.values())[0]
        n_target = sum(first_mask)
        n_total = len(first_mask)
        print(f"  MEMIT: residual norm = {r_norm:.4f}, facts={n_target}, total_keys={n_total}")
        weight = self.backend.get_layer_mlp_weight(critical_layer, self.target_module)
        if weight is not None:
            print(f"  MEMIT: key shape = {first_keys.shape}, weight shape = {weight.shape}")
        cov_mode = "covariance" if self._cov_diagonal else "identity"
        print(f"  MEMIT: regularization = {cov_mode}, lambda = {self.lambda_reg}")

        # Step 4: Distribute updates across layers (from L_last backwards)
        # Per MEMIT paper: at each layer, only absorb a fraction of the remaining
        # residual (1/remaining_layers), so the edit is spread across all layers.
        # This prevents single-layer overload and improves robustness.
        layer_deltas = {}
        reversed_layers = list(reversed(target_layers))
        num_target_layers = len(reversed_layers)

        for step_idx, layer_idx in enumerate(reversed_layers):
            keys_matrix, target_mask = keys_per_layer[layer_idx]
            hidden_size = fact_residuals.shape[1]
            total_keys = keys_matrix.shape[0]

            # Distribute: this layer absorbs 1/remaining of the residual
            remaining_layers = num_target_layers - step_idx
            distributed_residual = fact_residuals / remaining_layers

            # Build expanded residual: [total_keys, hidden_size]
            # Target positions get the distributed residual, constraint positions get 0
            if self._backend_type == "mlx":
                expanded_residuals = mx.zeros((total_keys, hidden_size), dtype=keys_matrix.dtype)
                fact_idx = 0
                for i, is_target in enumerate(target_mask):
                    if is_target:
                        expanded_residuals[i] = distributed_residual[fact_idx]
                        fact_idx += 1
                mx.eval(expanded_residuals)
            else:
                expanded_residuals = torch.zeros(total_keys, hidden_size,
                                                  dtype=keys_matrix.dtype, device=keys_matrix.device)
                # Move residual to keys' device (multi-GPU: layers live on different GPUs)
                dist_res = distributed_residual.to(keys_matrix.device)
                fact_idx = 0
                for i, is_target in enumerate(target_mask):
                    if is_target:
                        expanded_residuals[i] = dist_res[fact_idx]
                        fact_idx += 1

            delta = self._compute_layer_delta(keys_matrix, expanded_residuals, layer_idx=layer_idx)
            if delta is None:
                continue

            if self._backend_type == "mlx":
                d_norm = mx.sqrt(mx.sum(delta * delta)).item()
            else:
                d_norm = torch.norm(delta).item()
            print(f"    Layer {layer_idx} (1/{remaining_layers} residual): "
                  f"delta shape={delta.shape}, |delta|={d_norm:.6f}")
            self._apply_delta(layer_idx, delta)
            layer_deltas[layer_idx] = delta

            # Update fact residuals: subtract this layer's contribution at target positions
            if layer_idx != target_layers[0]:
                target_keys = []
                for i, is_target in enumerate(target_mask):
                    if is_target:
                        target_keys.append(keys_matrix[i])
                if target_keys:
                    target_key_matrix = ops.stack(target_keys)
                    contribution = ops.matmul(target_key_matrix, ops.transpose(delta))
                    if contribution.shape == fact_residuals.shape:
                        # Move to same device as fact_residuals (multi-GPU)
                        if hasattr(contribution, 'device') and hasattr(fact_residuals, 'device'):
                            contribution = contribution.to(fact_residuals.device)
                        fact_residuals = fact_residuals - contribution
                        ops.eval(fact_residuals)

        # Create edit record
        edit = MemitEdit(
            edit_id=str(uuid.uuid4())[:8],
            facts=facts,
            layer_deltas=layer_deltas,
            layer_indices=target_layers,
            key_vectors=keys_per_layer,
            timestamp=time.time(),
        )

        self._active_edits.append(edit)
        self.ledger.record_edit(edit)

        # Persist deltas to disk for reload across restarts
        self.ledger.save_deltas(edit.edit_id, layer_deltas)

        return edit

    def inject_fact(self, fact: FactTriple) -> Optional[MemitEdit]:
        """Convenience: wraps inject_facts([fact]) for single-fact use."""
        return self.inject_facts([fact])

    def revert_edit(self, edit: MemitEdit):
        """Subtract the currently-applied portion of delta from weights.

        Respects edit.scale — only removes (scale * delta), not the full delta.
        """
        ops = self._ops
        for layer_idx, delta in edit.layer_deltas.items():
            scaled = delta if edit.scale == 1.0 else self._scale_tensor(delta, edit.scale)
            self._apply_delta(layer_idx, ops.negate(scaled))

        self._active_edits = [e for e in self._active_edits if e.edit_id != edit.edit_id]

    def revert_all_active(self) -> int:
        """Revert all unconsolidated edits. Returns count of edits reverted."""
        count = len(self._active_edits)
        for edit in list(self._active_edits):
            self.revert_edit(edit)
        return count

    def scale_edit(self, edit: MemitEdit, new_scale: float):
        """Change the applied fraction of an edit's delta in the weights.

        The original full delta is always preserved in edit.layer_deltas.
        Only the applied portion changes: applies (new_scale - old_scale) * delta.
        """
        scale_diff = new_scale - edit.scale
        if abs(scale_diff) < 1e-8:
            return
        for layer_idx, delta in edit.layer_deltas.items():
            self._apply_delta(layer_idx, self._scale_tensor(delta, scale_diff))
        edit.scale = new_scale
        self.ledger.update_scale(edit.edit_id, new_scale, edit.consolidation_stage)

    def snapshot_target_weights(self) -> Dict[int, object]:
        """Copy target layer weights for later restoration.

        Returns:
            Dict of layer_idx -> weight tensor copy
        """
        snapshot = {}
        num_layers = self.backend.get_num_layers()
        target_layers = [l for l in self.target_layers if l < num_layers]
        for layer_idx in target_layers:
            weight = self.backend.get_layer_mlp_weight(layer_idx, self.target_module)
            if weight is not None:
                if self._backend_type == "mlx":
                    import mlx.core as mx
                    snapshot[layer_idx] = mx.array(weight)  # copy
                    mx.eval(snapshot[layer_idx])
                else:
                    snapshot[layer_idx] = weight.detach().clone()
        return snapshot

    def restore_target_weights(self, snapshot: Dict[int, object]):
        """Replace target layer weights from a snapshot. Exact pre-snapshot state."""
        for layer_idx, weight in snapshot.items():
            self.backend.set_layer_mlp_weight(layer_idx, self.target_module, weight)

    def reload_persisted_edits(self):
        """Re-apply all active MEMIT edits from persisted deltas after model reload.

        Called after backend.load() to restore MEMIT state across restarts.
        """
        active = self.ledger.get_active_edits()
        if not active:
            return

        reloaded = 0
        for edit_dict in active:
            edit_id = edit_dict["edit_id"]
            scale = edit_dict.get("scale", 1.0)
            if scale == 0.0:
                continue

            np_deltas = self.ledger.load_deltas(edit_id)
            if not np_deltas:
                continue

            # Convert numpy arrays to framework tensors and apply
            layer_deltas = {}
            for layer_idx, np_arr in np_deltas.items():
                if self._backend_type == "mlx":
                    import mlx.core as mx
                    tensor = mx.array(np_arr)
                else:
                    import torch
                    tensor = torch.from_numpy(np_arr)
                layer_deltas[layer_idx] = tensor
                self._apply_delta(layer_idx, self._scale_tensor(tensor, scale))

            # Reconstruct in-memory MemitEdit
            facts = [FactTriple.from_dict(f) for f in edit_dict["facts"]]
            edit = MemitEdit(
                edit_id=edit_id,
                facts=facts,
                layer_deltas=layer_deltas,
                layer_indices=edit_dict.get("layer_indices", []),
                timestamp=edit_dict.get("timestamp", 0),
                consolidated=edit_dict.get("consolidated", False),
                scale=scale,
                consolidation_stage=edit_dict.get("consolidation_stage", 0),
            )
            self._active_edits.append(edit)
            reloaded += 1

        if reloaded > 0:
            print(f"  MEMIT: reloaded {reloaded} persisted edit(s) from disk")

    def test_recall(self, fact: FactTriple, raw: bool = False) -> Tuple[bool, str]:
        """Test if the model can recall a fact.

        Args:
            fact: The fact triple to test
            raw: If True, use raw completion (matches MEMIT's edit pathway).
                 If False, use chat template (tests LoRA generalization).

        Returns:
            (passed, response_text)
        """
        if raw:
            prompt = fact.to_prompt()
            response = self.backend.generate(prompt, max_tokens=30, temperature=0.1)
        else:
            question = fact.to_question()
            prompt_messages = [{"role": "user", "content": question}]
            prompt = self.backend.apply_chat_template(prompt_messages)
            response = self.backend.generate(prompt, max_tokens=100, temperature=0.1)

        if response is None:
            response = ""
        passed = fact.object.lower() in response.lower()
        return passed, response

    def get_active_edit_count(self) -> int:
        """Return count of active in-memory edits."""
        return len(self._active_edits)

    def get_active_fact_count(self) -> int:
        """Return total active facts across all in-memory edits."""
        return sum(len(e.facts) for e in self._active_edits)

    # --- Internal methods ---

    def _compute_keys(self, facts: List[FactTriple], layer_idx: int):
        """Compute key matrix K at one layer with null-space constraints.

        Returns keys for ALL token positions (not just the target), with
        null-space flags indicating which rows are target keys vs constraint keys.
        The constraint keys enforce that the weight update doesn't affect
        non-target positions (preventing collateral damage).

        Also includes target keys from previous active MEMIT edits as additional
        null-space constraints — this prevents new edits from overwriting
        previously injected facts.

        Key = intermediate MLP activation (input to down_proj) = silu(gate_proj(x)) * up_proj(x).

        Returns:
            Tuple of (keys_matrix, target_mask):
              keys_matrix: [total_positions, intermediate_size] — all positions
              target_mask: [total_positions] — boolean, True for target positions
        """
        ops = self._ops
        all_keys = []
        target_mask = []

        for fact in facts:
            prompt_text = fact.to_prompt()
            tokens = self.backend.tokenizer.encode(prompt_text)
            input_ids = ops.to_device(ops.make_input_ids(tokens), self.backend)
            seq_len = len(tokens)

            # Get MLP input at target layer
            _, mlp_input, _ = self.backend.forward_to_layer(input_ids, layer_idx)

            # Compute intermediate activation (input to down_proj)
            intermediate = self.backend.compute_mlp_intermediate(mlp_input, layer_idx)

            # Include ALL positions as keys
            for pos in range(seq_len):
                key = intermediate[0, pos, :]  # [intermediate_size]
                all_keys.append(key)
                target_mask.append(pos == seq_len - 1)  # only last pos is target

        # Add previous edits' target keys as null-space constraints.
        # This prevents new weight updates from overwriting previously injected facts.
        # We recompute keys from the current model state (post-edit) for accuracy.
        prev_constraint_count = 0
        for prev_edit in self._active_edits:
            for prev_fact in prev_edit.facts:
                prompt_text = prev_fact.to_prompt()
                tokens = self.backend.tokenizer.encode(prompt_text)
                input_ids = ops.to_device(ops.make_input_ids(tokens), self.backend)
                seq_len = len(tokens)

                _, mlp_input, _ = self.backend.forward_to_layer(input_ids, layer_idx)
                intermediate = self.backend.compute_mlp_intermediate(mlp_input, layer_idx)

                # Only add the target position (last token) as constraint
                key = intermediate[0, seq_len - 1, :]
                all_keys.append(key)
                target_mask.append(False)  # constraint, not target
                prev_constraint_count += 1

        if prev_constraint_count > 0:
            print(f"      Layer {layer_idx}: {prev_constraint_count} previous-edit constraints added")

        keys_matrix = ops.stack(all_keys)  # [total_positions, intermediate_size]
        ops.eval(keys_matrix)
        return keys_matrix, target_mask

    def _compute_target_values(self, facts: List[FactTriple], critical_layer: int):
        """Compute target values via gradient-based optimization.

        For each fact, optimizes a delta vector that, when added to the MLP output
        at the subject's last token at the critical layer, causes the model to
        predict the target token. Uses the ROME v* optimization approach.
        """
        if self._backend_type == "torch":
            return self._compute_target_values_torch(facts, critical_layer)
        return self._compute_target_values_mlx(facts, critical_layer)

    def _compute_target_values_mlx(self, facts: List[FactTriple], critical_layer: int):
        """MLX implementation: optimize v* using autodiff through remaining layers.

        For each fact, finds a delta to the hidden state at the critical layer's
        last token position such that the model predicts the target token.
        Uses gradient descent on: -log P(target) + KL_factor * KL(baseline || modified).
        """
        import mlx.core as mx

        ops = self._ops
        targets = []
        v_lr = self.v_lr
        v_steps = self.v_steps
        kl_factor = self.v_kl_factor

        for fi, fact in enumerate(facts):
            prompt_text = fact.to_prompt()
            target_text = fact.to_target()
            prompt_tokens = self.backend.tokenizer.encode(prompt_text)
            full_tokens = self.backend.tokenizer.encode(prompt_text + target_text)

            input_ids = mx.array([prompt_tokens])
            seq_len = len(prompt_tokens)
            last_pos = seq_len - 1

            # Get hidden state at critical layer (residual stream after layer)
            hidden, _, mlp_output = self.backend.forward_to_layer(input_ids, critical_layer)
            mx.eval(hidden, mlp_output)

            v_current = mlp_output[0, last_pos, :]

            # Target token(s) that come after the prompt
            target_token_ids = full_tokens[len(prompt_tokens):]
            if not target_token_ids:
                targets.append(v_current)
                continue
            target_token_id = target_token_ids[0]

            # Causal attention mask for the prompt sequence
            # Use -1e4 instead of -inf: -inf causes NaN in softmax gradient
            # during backward pass. -1e4 is numerically equivalent for masking
            # (exp(-1e4) ≈ 0) but keeps gradients finite.
            mask = None
            if seq_len > 1:
                indices = mx.arange(seq_len)
                mask = (indices[:, None] < indices[None, :]).astype(hidden.dtype) * mx.array(-1e4, dtype=hidden.dtype)

            # Baseline logits (for KL constraint) — model's current output
            baseline_logits = self.backend.forward_from_layer(hidden, critical_layer + 1, mask=mask)
            baseline_lp = baseline_logits[0, last_pos, :]
            baseline_log_probs = baseline_lp - mx.logsumexp(baseline_lp, keepdims=True)
            mx.eval(baseline_log_probs)

            # Position mask: 1.0 only at last_pos, shape [1, seq_len, 1]
            pos_mask = (mx.arange(seq_len) == last_pos).astype(hidden.dtype).reshape(1, seq_len, 1)
            mx.eval(pos_mask)

            hidden_detached = mx.stop_gradient(hidden)

            # Check baseline probability of target token
            baseline_p_target = mx.exp(baseline_log_probs[target_token_id]).item()

            def loss_fn(delta):
                # Add delta only at last_pos in the residual stream
                h_modified = hidden_detached + pos_mask * delta.reshape(1, 1, -1)
                logits = self.backend.forward_from_layer(h_modified, critical_layer + 1, mask=mask)
                lp = logits[0, last_pos, :]
                log_probs = lp - mx.logsumexp(lp, keepdims=True)

                # Negative log prob of target token
                nll = -log_probs[target_token_id]

                # KL divergence from baseline (forward KL)
                kl = mx.sum(mx.exp(baseline_log_probs) * (baseline_log_probs - log_probs))

                return nll + kl_factor * kl

            grad_fn = mx.grad(loss_fn)

            delta = mx.zeros_like(v_current)
            mx.eval(delta)

            # Optimize delta to maximize P(target_token)
            for step in range(v_steps):
                g = grad_fn(delta)
                mx.eval(g)
                delta = delta - v_lr * g
                mx.eval(delta)

            # Compute final probability after optimization
            h_final = hidden_detached + pos_mask * delta.reshape(1, 1, -1)
            final_logits = self.backend.forward_from_layer(h_final, critical_layer + 1, mask=mask)
            final_lp = final_logits[0, last_pos, :]
            final_log_probs = final_lp - mx.logsumexp(final_lp, keepdims=True)
            final_p_target = mx.exp(final_log_probs[target_token_id]).item()

            delta_norm = mx.sqrt(mx.sum(delta * delta)).item()
            print(f"    v* opt fact {fi}: '{fact.subject} {fact.relation} → {fact.object}' "
                  f"token={target_token_id} "
                  f"P(target): {baseline_p_target:.4f} → {final_p_target:.4f}  "
                  f"|delta|={delta_norm:.4f}")

            v_star = v_current + delta
            mx.eval(v_star)
            targets.append(v_star)

        target_matrix = ops.stack(targets)
        ops.eval(target_matrix)
        return target_matrix

    def _compute_target_values_torch(self, facts: List[FactTriple], critical_layer: int):
        """PyTorch implementation: optimize v* using autodiff through remaining layers.

        Same algorithm as MLX version but using torch.autograd.
        """
        import torch
        import torch.nn.functional as F

        targets = []
        v_lr = self.v_lr
        v_steps = self.v_steps
        kl_factor = self.v_kl_factor
        # Use the device of the critical layer (handles multi-GPU device_map)
        critical = self.backend.model.model.layers[critical_layer]
        device = next(critical.parameters()).device

        for fi, fact in enumerate(facts):
            prompt_text = fact.to_prompt()
            target_text = fact.to_target()
            prompt_tokens = self.backend.tokenizer.encode(prompt_text)
            full_tokens = self.backend.tokenizer.encode(prompt_text + target_text)

            input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
            seq_len = len(prompt_tokens)
            last_pos = seq_len - 1

            # Get hidden state at critical layer
            with torch.no_grad():
                hidden, _, mlp_output = self.backend.forward_to_layer(input_ids, critical_layer)

            v_current = mlp_output[0, last_pos, :].detach().clone()

            # Target token(s)
            target_token_ids = full_tokens[len(prompt_tokens):]
            if not target_token_ids:
                targets.append(v_current)
                continue
            target_token_id = target_token_ids[0]

            # Baseline logits (for KL constraint)
            with torch.no_grad():
                baseline_logits = self._forward_from_layer_torch(
                    hidden.detach(), critical_layer + 1)
                baseline_lp = baseline_logits[0, last_pos, :]
                baseline_log_probs = baseline_lp - torch.logsumexp(baseline_lp, dim=0, keepdim=True)

            baseline_p_target = torch.exp(baseline_log_probs[target_token_id]).item()

            # Position mask: 1.0 only at last_pos
            pos_mask = torch.zeros(1, seq_len, 1, device=device, dtype=hidden.dtype)
            pos_mask[0, last_pos, 0] = 1.0

            hidden_detached = hidden.detach()

            # Optimize delta
            delta = torch.zeros_like(v_current, requires_grad=True)
            optimizer = torch.optim.SGD([delta], lr=v_lr)

            for step in range(v_steps):
                optimizer.zero_grad()
                h_modified = hidden_detached + pos_mask * delta.unsqueeze(0).unsqueeze(0)
                logits = self._forward_from_layer_torch(h_modified, critical_layer + 1)
                lp = logits[0, last_pos, :]
                log_probs = lp - torch.logsumexp(lp, dim=0, keepdim=True)

                nll = -log_probs[target_token_id]
                kl = torch.sum(torch.exp(baseline_log_probs) * (baseline_log_probs - log_probs))
                loss = nll + kl_factor * kl
                loss.backward()
                optimizer.step()

            # Final probability
            with torch.no_grad():
                h_final = hidden_detached + pos_mask * delta.detach().unsqueeze(0).unsqueeze(0)
                final_logits = self._forward_from_layer_torch(h_final, critical_layer + 1)
                final_lp = final_logits[0, last_pos, :]
                final_log_probs = final_lp - torch.logsumexp(final_lp, dim=0, keepdim=True)
                final_p_target = torch.exp(final_log_probs[target_token_id]).item()

            delta_norm = torch.norm(delta.detach()).item()
            print(f"    v* opt fact {fi}: '{fact.subject} {fact.relation} → {fact.object}' "
                  f"token={target_token_id} "
                  f"P(target): {baseline_p_target:.4f} → {final_p_target:.4f}  "
                  f"|delta|={delta_norm:.4f}")

            v_star = (v_current + delta.detach()).detach()
            targets.append(v_star)

        target_matrix = torch.stack(targets)
        return target_matrix

    def _forward_from_layer_torch(self, hidden_states, start_layer):
        """Forward from a layer to logits WITH gradients enabled.

        Unlike backend.forward_from_layer() which uses no_grad, this
        allows gradients to flow through for v* optimization.
        Handles multi-GPU device_map by moving tensors between devices.
        """
        import torch
        h = hidden_states
        seq_len = h.shape[1]
        position_ids = torch.arange(seq_len, device=h.device).unsqueeze(0)
        position_embeddings = self.backend.model.model.rotary_emb(h, position_ids)
        for i in range(start_layer, len(self.backend.model.model.layers)):
            layer = self.backend.model.model.layers[i]
            layer_device = next(layer.parameters()).device
            h = h.to(layer_device)
            pos_emb = tuple(p.to(layer_device) for p in position_embeddings)
            layer_out = layer(h, position_embeddings=pos_emb)
            h = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        norm_device = next(self.backend.model.model.norm.parameters()).device
        h = self.backend.model.model.norm(h.to(norm_device))
        head_device = next(self.backend.model.lm_head.parameters()).device
        logits = self.backend.model.lm_head(h.to(head_device))
        return logits

    def _compute_current_values(self, facts: List[FactTriple], layer_idx: int):
        """Compute current MLP output values at one layer for just the prompt."""
        ops = self._ops
        values = []
        for fact in facts:
            prompt_text = fact.to_prompt()
            tokens = self.backend.tokenizer.encode(prompt_text)
            input_ids = ops.to_device(ops.make_input_ids(tokens), self.backend)

            _, _, mlp_output = self.backend.forward_to_layer(input_ids, layer_idx)

            # Value at last token position
            value = mlp_output[0, -1, :]  # [hidden_size]
            values.append(value)

        value_matrix = ops.stack(values)  # [num_facts, hidden_size]
        ops.eval(value_matrix)
        return value_matrix

    def _compute_layer_delta(self, keys, residuals, layer_idx=None):
        """Compute weight delta for one layer.

        When covariance estimates are available, uses the MEMIT paper's proper
        regularization via the Woodbury identity:
          ΔW = R^T @ S^{-1} @ K_w
        where K̃ = K/(√λ·σ), K_w = K/(λ·σ²), S = I + K̃@K̃^T

        This keeps the inversion in [N,N] space (small) while incorporating
        per-dimension covariance regularization that protects the model's
        heavily-used activation dimensions from corruption.

        Fallback (no covariance): ΔW = R^T @ (KK^T + λI)^{-1} @ K

        Shapes:
          K: [num_keys, intermediate_size]
          R: [num_keys, hidden_size]
          ΔW: [hidden_size, intermediate_size]  (matches down_proj weight)
        """
        ops = self._ops
        num_keys = keys.shape[0]

        # Check if covariance is available for this layer
        has_cov = (layer_idx is not None and layer_idx in self._cov_diagonal)

        if has_cov:
            return self._compute_layer_delta_covariance(keys, residuals, layer_idx)
        else:
            return self._compute_layer_delta_identity(keys, residuals)

    def _compute_layer_delta_covariance(self, keys, residuals, layer_idx):
        """Covariance-regularized delta using Woodbury identity.

        From the MEMIT paper: ΔW = R @ K^T @ (λC₀ + K^T @ K)^{-1}
        Using Woodbury to keep inversion in [N,N] space:
          ΔW = R^T @ S^{-1} @ K_w
        where:
          σ² = diagonal covariance at this layer
          K̃ = K / (√λ · σ)  — whitened keys
          K_w = K / (λ · σ²) — doubly-scaled keys
          S = I + K̃ @ K̃^T   — [N, N] matrix
        """
        ops = self._ops
        num_keys = keys.shape[0]
        cov_diag = self._cov_diagonal[layer_idx]  # [intermediate_size]

        if self._backend_type == "mlx":
            import mlx.core as mx

            # Compute everything in float32 for numerical stability
            # (keys may be float16, covariance is float32)
            keys_f32 = keys.astype(mx.float32)
            residuals_f32 = residuals.astype(mx.float32)
            cov_f32 = cov_diag.astype(mx.float32)

            # Compute scaling factors
            # inv_cov_sqrt = 1 / (sqrt(λ) * σ) for whitening
            # inv_cov = 1 / (λ * σ²) for the doubly-scaled keys
            lambda_cov = self.lambda_reg * cov_f32  # [d_in]
            inv_cov_sqrt = 1.0 / mx.sqrt(lambda_cov + 1e-8)  # [d_in]
            inv_cov = 1.0 / (lambda_cov + 1e-8)  # [d_in]
            mx.eval(inv_cov_sqrt, inv_cov)

            # Whitened keys: K̃ = K * inv_cov_sqrt (broadcast)
            K_tilde = keys_f32 * inv_cov_sqrt  # [N, d_in]
            mx.eval(K_tilde)

            # Doubly-scaled keys: K_w = K * inv_cov (broadcast)
            K_w = keys_f32 * inv_cov  # [N, d_in]
            mx.eval(K_w)

            # S = I + K̃ @ K̃^T  [N, N]
            S = mx.eye(num_keys, dtype=mx.float32) + K_tilde @ K_tilde.T
            mx.eval(S)

            # Invert S (small [N, N] matrix)
            try:
                S_inv = mx.linalg.inv(S, stream=mx.cpu)
                mx.eval(S_inv)
            except Exception as e:
                print(f"    covariance delta: inv failed ({e}), adding extra reg")
                try:
                    S_reg = S + 0.1 * mx.eye(num_keys, dtype=mx.float32)
                    S_inv = mx.linalg.inv(S_reg, stream=mx.cpu)
                    mx.eval(S_inv)
                except Exception as e2:
                    print(f"    covariance delta: inv failed again ({e2}), falling back to identity")
                    return self._compute_layer_delta_identity(keys, residuals)

            # ΔW = R^T @ S_inv @ K_w  [hidden_size, intermediate_size]
            delta = residuals_f32.T @ S_inv @ K_w
            # Convert back to original dtype for weight update
            delta = delta.astype(keys.dtype)
            mx.eval(delta)
            return delta

        else:
            import torch

            orig_dtype = keys.dtype
            keys_f32 = keys.float()
            residuals_f32 = residuals.float()
            cov_f32 = cov_diag.to(device=keys.device).float()

            lambda_cov = self.lambda_reg * cov_f32
            inv_cov_sqrt = 1.0 / torch.sqrt(lambda_cov + 1e-8)
            inv_cov = 1.0 / (lambda_cov + 1e-8)

            K_tilde = keys_f32 * inv_cov_sqrt
            K_w = keys_f32 * inv_cov

            S = torch.eye(num_keys, dtype=torch.float32, device=keys.device) + K_tilde @ K_tilde.T

            try:
                S_inv = torch.linalg.inv(S)
            except Exception as e:
                print(f"    covariance delta: inv failed ({e}), falling back to identity")
                return self._compute_layer_delta_identity(keys, residuals)

            delta = residuals_f32.T @ S_inv @ K_w
            return delta.to(orig_dtype)

    def _compute_layer_delta_identity(self, keys, residuals):
        """Fallback: identity-regularized delta (original formula).

        ΔW = R^T @ (K @ K^T + λI)^{-1} @ K
        """
        ops = self._ops
        num_keys = keys.shape[0]

        KKT = ops.matmul(keys, ops.transpose(keys))
        ops.eval(KKT)

        if self._backend_type == "mlx":
            import mlx.core as mx
            reg_eye = mx.eye(num_keys, dtype=KKT.dtype)
        else:
            import torch
            reg_eye = torch.eye(num_keys, dtype=torch.float32, device=KKT.device)
        KKT_reg = KKT + self.lambda_reg * reg_eye
        ops.eval(KKT_reg)

        try:
            if self._backend_type == "mlx":
                KKT_f32 = KKT_reg.astype(mx.float32)
                KKT_inv = mx.linalg.inv(KKT_f32, stream=mx.cpu).astype(KKT_reg.dtype)
            else:
                KKT_inv = torch.linalg.inv(KKT_reg.float()).to(KKT_reg.dtype)
            ops.eval(KKT_inv)
        except Exception as e:
            print(f"    identity delta: inv failed ({e}), trying extra reg")
            try:
                extra_reg = 0.1 * reg_eye
                if self._backend_type == "mlx":
                    KKT_f32 = (KKT_reg + extra_reg).astype(mx.float32)
                    KKT_inv = mx.linalg.inv(KKT_f32, stream=mx.cpu).astype(KKT_reg.dtype)
                else:
                    KKT_inv = torch.linalg.inv((KKT_reg + extra_reg).float()).to(KKT_reg.dtype)
                ops.eval(KKT_inv)
            except Exception as e2:
                print(f"    identity delta: inv failed again ({e2}), returning None")
                return None

        delta = ops.matmul(ops.matmul(ops.transpose(residuals), KKT_inv), keys)
        ops.eval(delta)
        return delta

    def _apply_delta(self, layer_idx: int, delta_weight):
        """Add delta to MLP down_proj weight at the given layer."""
        current_weight = self.backend.get_layer_mlp_weight(layer_idx, self.target_module)
        if current_weight is None:
            return

        # Ensure shapes match — delta might need transposing
        if delta_weight.shape != current_weight.shape:
            t_delta = self._ops.transpose(delta_weight)
            if t_delta.shape == current_weight.shape:
                delta_weight = t_delta
            else:
                return

        # Ensure same device and dtype
        if self._backend_type == "mlx":
            new_weight = current_weight + delta_weight.astype(current_weight.dtype)
        else:
            if hasattr(delta_weight, 'device') and delta_weight.device != current_weight.device:
                delta_weight = delta_weight.to(current_weight.device)
            new_weight = current_weight + delta_weight.to(current_weight.dtype)
        self.backend.set_layer_mlp_weight(layer_idx, self.target_module, new_weight)

    def _scale_tensor(self, tensor, scale: float):
        """Multiply a tensor by a scalar, handling both MLX and PyTorch."""
        if self._backend_type == "mlx":
            import mlx.core as mx
            result = tensor * mx.array(scale, dtype=tensor.dtype)
            mx.eval(result)
            return result
        else:
            return tensor * scale

    def _build_prompt_for_fact(self, fact: FactTriple) -> str:
        """Convert a FactTriple to a natural language prompt."""
        return fact.to_prompt()
