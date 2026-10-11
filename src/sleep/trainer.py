"""Sleep trainer — LoRA training orchestrator for consolidation.

Prepares training data from QAPair facts (native chat format) and
delegates LoRA training + fusing to the backend.

Training uses the canonical question plus the fact's TRAINING paraphrases,
paired round-robin with its answer variants. The HELD-OUT paraphrases go to
valid.jsonl only; prepare_training_data drops any training row that matches a
held-out phrasing, so no held-out line ever reaches train.jsonl.
"""

import json
import random
from pathlib import Path
from typing import List, Optional

from src.sleep.paraphrase import ParaphraseGenerator
from src.sleep.probe import build_system_prompt
from src.sleep.recall_match import normalize


def _chat_example(question: str, answer: str) -> dict:
    return {"messages": [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}]}


def _user_turn(row: dict) -> str:
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def build_training_examples(qa) -> List[dict]:
    """Canonical question + training paraphrases, round-robin over answer variants."""
    questions = [qa.question]
    seen = {normalize(qa.question)}
    for p in getattr(qa, "paraphrases", []) or []:
        if normalize(p) not in seen:
            seen.add(normalize(p))
            questions.append(p)
    answers = list(getattr(qa, "answer_variants", []) or []) or [qa.answer]
    return [_chat_example(q, answers[i % len(answers)]) for i, q in enumerate(questions)]


def build_validation_examples(qa) -> List[dict]:
    """Held-out paraphrases with the canonical answer."""
    return [_chat_example(q, qa.answer) for q in (getattr(qa, "held_out", []) or [])]


class SleepTrainer:
    """Orchestrates LoRA training and fusing during sleep consolidation."""

    def __init__(self, config, backend):
        self.config = config
        self.backend = backend

        lora_cfg = config.get("lora", {}) or {}
        self.num_layers = lora_cfg.get("num_layers", 8)
        self.learning_rate = lora_cfg.get("learning_rate", 1e-4)
        self.iters_per_fact = lora_cfg.get("iters_per_fact", 10)
        self.batch_size = lora_cfg.get("batch_size", 1)
        # Contrast rows per fact: the same question about another person,
        # answered "I don't know". Teaches the subject binding the bleed gate
        # tests for. 0 disables.
        self.contrast_per_fact = int(lora_cfg.get("contrast_per_fact", 2))
        # Deployment-time context in training rows: a system prompt carrying up
        # to `context_facts` OTHER active facts on `context_fraction` of the
        # rows. Graduation probes and the app both put the still-in-prompt
        # facts there; a model trained without that learns "the prompt is all
        # I know" and denies what its weights hold (notes/134).
        self.context_facts = int(lora_cfg.get("context_facts", 3))
        self.context_fraction = float(lora_cfg.get("context_fraction", 0.5))
        self.system_prompt = (getattr(config, "context", None) or {}).get("system_prompt", "")

    def _with_context(self, row: dict, qa, context_qa, rng) -> dict:
        """Prepend a system message: the base prompt, plus other facts on a
        `context_fraction` share of rows. The row's own fact never appears."""
        others = []
        if context_qa and self.context_facts > 0 and rng.random() < self.context_fraction:
            pool = [f for f in context_qa
                    if normalize(f.question) != normalize(qa.question)
                    and normalize(f.answer) != normalize(qa.answer)]
            k = min(len(pool), rng.randint(1, self.context_facts))
            others = rng.sample(pool, k) if k else []
        system = build_system_prompt(self.system_prompt, others)
        return {"messages": [{"role": "system", "content": system}] + row["messages"]}

    def prepare_training_data(self, qa_pairs, output_dir, weighted=False,
                              avoid_questions=(), context_qa=None) -> Path:
        """Write train.jsonl / valid.jsonl for a list of QAPairs.

        weighted=True repeats a fact's examples by priority
        (1.0 → 3x, 0.5 → 2x, 0.0 → 1x) and shuffles.

        Each fact also contributes `contrast_per_fact` neighbor questions with
        a don't-know answer (see ParaphraseGenerator.contrast_examples).
        `avoid_questions` lists phrasings that must never become a contrast
        row: every ledger question and every bleed-gate probe.

        context_qa (the facts currently carried in the prompt) switches on
        system messages: every row gets the base system prompt and a share of
        rows also get a few of the OTHER facts, as the probe would show them.

        valid.jsonl holds the held-out paraphrases. If no fact has any, it
        falls back to a copy of the training set (mlx_lm requires the file).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(len(qa_pairs) * 7919 + sum(len(qa.question) for qa in qa_pairs))
        avoid = list(avoid_questions) + [qa.question for qa in qa_pairs]
        train_rows: List[dict] = []
        valid_rows: List[dict] = []
        n_contrast = 0
        for qa in qa_pairs:
            examples = build_training_examples(qa)
            reps = 1
            if weighted:
                priority = getattr(qa, "priority", 0.5)
                reps = max(1, round(1 + priority * 2))
            rows = [dict(e) for _ in range(reps) for e in examples]
            contrast = ParaphraseGenerator.contrast_examples(
                qa, self.contrast_per_fact, avoid_questions=avoid)
            avoid.extend(q for q, _ in contrast)
            rows.extend(_chat_example(q, a) for q, a in contrast)
            n_contrast += len(contrast)
            if context_qa is not None:
                rows = [self._with_context(r, qa, context_qa, rng) for r in rows]
            train_rows.extend(rows)
            valid_rows.extend(build_validation_examples(qa))

        if weighted:
            random.shuffle(train_rows)

        # Invariant: no held-out phrasing (of ANY fact) in the training file.
        # A collision can only come from identical rephrasings handed to two
        # facts; the training row is dropped, the held-out set is kept intact.
        held_out = {normalize(q) for qa in qa_pairs for q in (getattr(qa, "held_out", []) or [])}
        kept = [r for r in train_rows if normalize(_user_turn(r)) not in held_out]
        if len(kept) != len(train_rows):
            print(f"        WARNING: dropped {len(train_rows) - len(kept)} training row(s) that "
                  f"matched a held-out phrasing")
            train_rows = kept
        if not train_rows:
            raise ValueError("no training rows left after removing held-out phrasings")

        if not valid_rows:
            valid_rows = list(train_rows)

        (output_dir / "train.jsonl").write_text(
            "\n".join(json.dumps(r) for r in train_rows) + "\n")
        (output_dir / "valid.jsonl").write_text(
            "\n".join(json.dumps(r) for r in valid_rows) + "\n")

        n_held = sum(len(getattr(qa, "held_out", []) or []) for qa in qa_pairs)
        print(f"        Training data: {len(qa_pairs)} facts → {len(train_rows)} examples"
              f"{' (priority-weighted)' if weighted else ''} incl. {n_contrast} contrast, "
              f"{n_held} held-out phrasing(s) in valid.jsonl")
        return output_dir

    def prepare_weighted_training_data(self, qa_pairs, output_dir) -> Path:
        """Priority-weighted training data (kept for callers that use this name)."""
        return self.prepare_training_data(qa_pairs, output_dir, weighted=True)

    def compute_iters(self, qa_pairs, iters=None) -> int:
        if iters is not None:
            return int(iters)
        return max(len(qa_pairs) * self.iters_per_fact, 20)

    def train_and_fuse(self, qa_pairs, cycle_id, save_dir, weighted=False,
                       iters=None, avoid_questions=(), context_qa=None) -> Optional[str]:
        """Train LoRA on QAPairs and fuse into the model.

        Args:
            qa_pairs: List of QAPair objects to train on
            cycle_id: Sleep cycle identifier (for naming)
            save_dir: Directory for fused model output
            weighted: Use priority-weighted repetition
            iters: Override iteration count (default len(facts) * iters_per_fact)
            avoid_questions: Phrasings that must not become contrast rows
                (ledger questions, bleed-gate probes)
            context_qa: Facts currently carried in the prompt; enables
                system-prompt context in training rows

        Returns:
            Path to fused model directory, or None on failure.
        """
        save_dir = Path(save_dir)
        data_dir = save_dir / "training_data" / cycle_id
        adapter_path = save_dir / "adapters" / cycle_id
        fused_path = save_dir / "fused" / cycle_id

        # 1. Prepare training data
        self.prepare_training_data(qa_pairs, data_dir, weighted=weighted,
                                   avoid_questions=avoid_questions, context_qa=context_qa)

        # 2. Compute iterations
        iters = self.compute_iters(qa_pairs, iters)

        # 3. Train LoRA
        try:
            print(f"        Training LoRA ({iters} iters, {self.num_layers} layers)...")
            self.backend.train_lora(
                data_path=str(data_dir),
                adapter_path=str(adapter_path),
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                iters=iters,
                learning_rate=self.learning_rate,
            )
        except Exception as e:
            print(f"        LoRA training failed: {e}")
            return None

        # 4. Fuse adapter into model
        try:
            print(f"        Fusing adapter...")
            self.backend.fuse_adapter(
                adapter_path=str(adapter_path),
                save_path=str(fused_path),
            )
        except Exception as e:
            print(f"        Fuse failed: {e}")
            return None

        print(f"        Fused model saved to {fused_path}")
        return str(fused_path)
