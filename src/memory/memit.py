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
    They are needed for reverting edits.
    """
    edit_id: str
    facts: List[FactTriple]
    layer_deltas: Dict[int, object] = field(default_factory=dict)  # layer_idx -> mx.array
    layer_indices: List[int] = field(default_factory=list)
    key_vectors: Dict[int, object] = field(default_factory=dict)  # layer_idx -> mx.array
    timestamp: float = field(default_factory=time.time)
    consolidated: bool = False

    def to_ledger_dict(self) -> dict:
        """Serialize metadata only (no arrays) for the ledger."""
        return {
            "edit_id": self.edit_id,
            "facts": [f.to_dict() for f in self.facts],
            "layer_indices": self.layer_indices,
            "timestamp": self.timestamp,
            "consolidated": self.consolidated,
        }


class EditLedger:
    """Persists MEMIT edit metadata to disk.

    Tracks which facts have been injected via MEMIT, which are still active,
    and which have been consolidated into LoRA during sleep.
    """

    def __init__(self, ledger_path: str):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._edits: List[dict] = []
        self.load()

    def record_edit(self, edit: MemitEdit):
        """Record a new MEMIT edit."""
        self._edits.append(edit.to_ledger_dict())
        self.save()

    def get_active_edits(self) -> List[dict]:
        """Return all non-consolidated edits."""
        return [e for e in self._edits if not e.get("consolidated", False)]

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


class MemitEngine:
    """Core MEMIT implementation for MLX.

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
        self.lambda_reg = memit_config.get("lambda_reg", 0.5)
        self.max_active_edits = memit_config.get("max_active_edits", 50)
        self.enabled = memit_config.get("enabled", True)
        self.target_module = memit_config.get("target_module", "down_proj")
        self.covariance_samples = memit_config.get("covariance_samples", 0)

        # In-memory store of active edits (with delta arrays for revert)
        self._active_edits: List[MemitEdit] = []

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

        import mlx.core as mx

        model = self.backend.model
        tokenizer = self.backend.tokenizer

        # Clamp target layers to actual model size
        num_layers = self.backend.get_num_layers()
        target_layers = [l for l in self.target_layers if l < num_layers]
        if not target_layers:
            return None

        critical_layer = target_layers[-1]  # L_last

        # Step 1: Compute key vectors at each target layer
        keys_per_layer = {}
        for layer_idx in target_layers:
            keys = self._compute_keys(facts, layer_idx)
            if keys is None:
                return None
            keys_per_layer[layer_idx] = keys

        # Step 2: Compute target values at critical layer
        target_values = self._compute_target_values(facts, critical_layer)
        if target_values is None:
            return None

        # Step 3: Compute current values at critical layer
        current_values = self._compute_current_values(facts, critical_layer)
        if current_values is None:
            return None

        # Initial residual = target - current
        residuals = target_values - current_values

        # Step 4: Distribute updates across layers (from L_last backwards)
        layer_deltas = {}
        for layer_idx in reversed(target_layers):
            keys = keys_per_layer[layer_idx]
            delta = self._compute_layer_delta(keys, residuals)
            if delta is None:
                continue

            # Apply delta
            self._apply_delta(layer_idx, delta)
            layer_deltas[layer_idx] = delta

            # Update residuals: subtract this layer's contribution
            # New residuals = old residuals - delta @ keys
            if layer_idx != target_layers[0]:
                contribution = delta @ keys.T
                # Only subtract if shapes match (they should for same batch)
                if contribution.shape == residuals.shape:
                    residuals = residuals - contribution
                    mx.eval(residuals)

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

        return edit

    def inject_fact(self, fact: FactTriple) -> Optional[MemitEdit]:
        """Convenience: wraps inject_facts([fact]) for single-fact use."""
        return self.inject_facts([fact])

    def revert_edit(self, edit: MemitEdit):
        """Subtract all delta matrices from weights, reversing the edit."""
        import mlx.core as mx

        for layer_idx, delta in edit.layer_deltas.items():
            self._apply_delta(layer_idx, -delta)

        self._active_edits = [e for e in self._active_edits if e.edit_id != edit.edit_id]

    def revert_all_active(self) -> int:
        """Revert all unconsolidated edits. Returns count of edits reverted."""
        count = len(self._active_edits)
        for edit in list(self._active_edits):
            self.revert_edit(edit)
        return count

    def test_recall(self, fact: FactTriple) -> Tuple[bool, str]:
        """Generate a question from the triple, check if response contains the object.

        Returns:
            (passed, response_text)
        """
        question = fact.to_question()
        prompt_messages = [{"role": "user", "content": question}]
        prompt = self.backend.apply_chat_template(prompt_messages)
        response = self.backend.generate(prompt, max_tokens=100, temperature=0.1)

        # Check if the object appears in the response
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
        """Compute key matrix K [num_facts x hidden_size] at one layer.

        Key = hidden state at subject's last token at MLP input of the target layer.
        """
        import mlx.core as mx

        keys = []
        for fact in facts:
            prompt_text = fact.to_prompt()
            tokens = self.backend.tokenizer.encode(prompt_text)
            input_ids = mx.array([tokens])

            # Get hidden state at target layer's MLP input
            hidden, mlp_input, _ = self.backend.forward_to_layer(input_ids, layer_idx)

            # Key = MLP input at last token position
            key = mlp_input[0, -1, :]  # [hidden_size]
            keys.append(key)

        keys_matrix = mx.stack(keys)  # [num_facts, hidden_size]
        mx.eval(keys_matrix)
        return keys_matrix

    def _compute_target_values(self, facts: List[FactTriple], critical_layer: int):
        """Compute target value matrix at the critical layer.

        Target = the MLP output representation at critical layer when running
        the full text (subject + relation + object).
        """
        import mlx.core as mx

        targets = []
        for fact in facts:
            full_text = fact.to_prompt() + fact.to_target()
            tokens = self.backend.tokenizer.encode(full_text)
            input_ids = mx.array([tokens])

            # Get MLP output at critical layer for the full text
            _, _, mlp_output = self.backend.forward_to_layer(input_ids, critical_layer)

            # Find subject's last token position
            subject_tokens = self.backend.tokenizer.encode(fact.to_prompt())
            subj_last_pos = len(subject_tokens) - 1

            target = mlp_output[0, subj_last_pos, :]  # [hidden_size]
            targets.append(target)

        target_matrix = mx.stack(targets)  # [num_facts, hidden_size]
        mx.eval(target_matrix)
        return target_matrix

    def _compute_current_values(self, facts: List[FactTriple], layer_idx: int):
        """Compute current MLP output values at one layer for just the prompt."""
        import mlx.core as mx

        values = []
        for fact in facts:
            prompt_text = fact.to_prompt()
            tokens = self.backend.tokenizer.encode(prompt_text)
            input_ids = mx.array([tokens])

            _, _, mlp_output = self.backend.forward_to_layer(input_ids, layer_idx)

            # Value at last token position
            value = mlp_output[0, -1, :]  # [hidden_size]
            values.append(value)

        value_matrix = mx.stack(values)  # [num_facts, hidden_size]
        mx.eval(value_matrix)
        return value_matrix

    def _compute_layer_delta(self, keys, residuals):
        """Compute weight delta for one layer using least-squares.

        ΔW = R @ K^T @ (K @ K^T + λI)^{-1}

        Where R = residuals matrix, K = key matrix.
        This finds the minimum-norm weight update that maps keys to residuals.
        """
        import mlx.core as mx

        num_facts = keys.shape[0]

        # K @ K^T: [num_facts x num_facts]
        KKT = keys @ keys.T

        # Regularize: KKT + λI
        reg = self.lambda_reg * mx.eye(num_facts)
        KKT_reg = KKT + reg

        # Solve: (K @ K^T + λI)^{-1}
        try:
            KKT_inv = mx.linalg.inv(KKT_reg)
        except Exception:
            # Fallback: use pseudo-inverse via SVD
            try:
                # Simple regularized pseudo-inverse
                KKT_inv = mx.linalg.inv(KKT_reg + 0.1 * mx.eye(num_facts))
            except Exception:
                return None

        # ΔW = R @ K^T @ (K @ K^T + λI)^{-1}
        # R: [num_facts, hidden_size], K: [num_facts, hidden_size]
        # We want ΔW: [hidden_size, hidden_size] (to add to down_proj weight)
        # Actually: ΔW = (KKT_inv @ R)^T @ K  — rearranged for correct shapes
        # Or equivalently: ΔW^T = K^T @ KKT_inv @ R
        # So ΔW = R^T @ KKT_inv^T @ K = R^T @ KKT_inv @ K (since KKT_inv is symmetric)
        delta = residuals.T @ KKT_inv @ keys  # [hidden_size, hidden_size]

        mx.eval(delta)
        return delta

    def _apply_delta(self, layer_idx: int, delta_weight):
        """Add delta to MLP down_proj weight at the given layer."""
        import mlx.core as mx

        current_weight = self.backend.get_layer_mlp_weight(layer_idx, self.target_module)
        if current_weight is None:
            return

        # Ensure shapes match — delta might need transposing
        if delta_weight.shape != current_weight.shape:
            if delta_weight.T.shape == current_weight.shape:
                delta_weight = delta_weight.T
            else:
                # Shape mismatch — skip this layer
                return

        new_weight = current_weight + delta_weight
        self.backend.set_layer_mlp_weight(layer_idx, self.target_module, new_weight)

    def _build_prompt_for_fact(self, fact: FactTriple) -> str:
        """Convert a FactTriple to a natural language prompt."""
        return fact.to_prompt()
