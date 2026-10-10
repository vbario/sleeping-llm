# Sleeping LLM

A local language model that forms persistent memories from conversation and consolidates them into its weights during sleep.

During **wake**, facts are extracted from the conversation as question–answer pairs, stored in a ledger, and injected into the system prompt, so the model can use them immediately. During **sleep**, a LoRA adapter is trained on those facts and fused into the base weights. A fact that the fused model can answer **from its weights alone**, when asked in words it was never trained on, *graduates*: it leaves the prompt and the weights carry it. Graduated facts are replayed and re-audited in later sleeps so they are not forgotten, every fuse must pass gates before it replaces the current model, and a changed fact replaces the old one instead of piling up next to it.

Inspired by [Complementary Learning Systems](https://en.wikipedia.org/wiki/Complementary_learning_systems): the prompt is fast, exact, capacity-limited memory; the weights are slow, high-capacity memory; sleep is the transfer between them.

> **Status.** This README describes the code on `main`. The MEMIT weight-editing system described in the published papers is no longer on the runtime path; see [Earlier versions](#earlier-versions-memit--lora) for where that code lives and the results it produced. No recall numbers have yet been measured for the current pipeline; see [Evaluation](#evaluation).

## Architecture

```
WAKE                                           SLEEP  (src/sleep/full_sleep.py)

  User ←→ Chat                                 1. Health check — baseline perplexity
     │                                         2. Curate — extract facts from unconsumed
     ▼                                            sessions, upsert into the ledger
  Fact extraction (Q&A pairs)                  3. Consolidate (src/sleep/consolidation.py)
     │                                              · replay sample of graduated facts
     ▼                                              · generate held-out paraphrases
  FactBuffer (RAM, surprise-gated)                  · train LoRA, fuse → candidate model
     │  consolidation moment                        · GATES: perplexity, bleed, regression
     ▼                                                  fail → reload last good model,
  FactLedger (ledger.json)                                     nothing changes, retry w/o culprit
     │  slot-aware upsert                      4. Graduate — ask each fact's held-out
     ▼                                            phrasings with the fact withheld;
  System prompt                                   pass → stage+1 (3 = graduated, leaves
  "Things you remember about the user:"           prompt); fail → stage−1. Re-audit a
     - The user's son is named Andre.             sample of graduated facts; retreat losers
     - ...                                     5. Validate — perplexity vs baseline
  (non-graduated facts only)
                                               Micro-sleep: background LoRA on high-priority
                                               facts, same runner, same gates.
                                               Nap: audit recent + graduated facts, no training.
```

### Memory tiers

| Tier | Where | Role |
|---|---|---|
| 0 | System prompt | Instant recall of every non-graduated fact. Rebuilt from the ledger each turn. |
| 1 | `FactBuffer` (RAM) | Facts extracted this session, waiting for a consolidation moment. Volatile by design. |
| 2 | `FactLedger` (`data/memit/ledger.json`, `graduated.json`) | Source of truth: every fact, its stage, training/audit history, paraphrases, supersede links. |
| 3 | Fused LoRA weights (`models/fused/`) | Long-term memory. Only graduated facts rely on it alone. |

### Fact lifecycle

Each fact is a `QAPair` (question, answer, value) with a stage 0–3.

1. **Paraphrases.** Before a fact is first trained, several rephrasings are generated (first-person template rewrites plus optional model rephrasings). Some are used for training; at least two are **held out** and only ever used for testing. A held-out line never reaches a training file (`src/sleep/paraphrase.py`, `src/sleep/trainer.py`).
2. **Training.** Every sleep trains on all non-graduated facts (canonical question + training paraphrases × answer variants, priority-weighted) **plus a replay sample of graduated facts**, chosen degraded-first then least-recently-verified (`FactLedger.get_replay_candidates`).
3. **Graduation test.** With the fact removed from the prompt (other facts stay), the held-out phrasings are asked. A response counts only if it states the value on word boundaries and is not hedged (`src/sleep/recall_match.py`, `src/sleep/probe.py`). Passing the configured fraction advances one stage; failing retreats one stage. At stage 3 the fact moves to `graduated.json` and leaves the prompt.
4. **Audit.** Every nap and every full sleep re-tests a sample of graduated facts. A graduated fact that fails returns to stage 2, re-enters the prompt and the training set, and its `degrade_count` is recorded.

### Gates and rollback

Every fuse, in full sleep and in micro-sleep, produces a *candidate* model. The candidate is loaded and three gates run (`src/sleep/gates.py`):

| Gate | Check | Culprit? |
|---|---|---|
| perplexity | relative increase on reference text vs the pre-sleep baseline ≤ `max_ppl_increase` | no |
| bleed | for each newly trained fact, neighbor questions (same question about a different person) answered before training must not now answer with the fact's value | yes |
| regression | a sample of graduated facts must still be recalled on the candidate | no |

If all pass, graduation tests run and the candidate is committed as the last known-good model; older fused models are pruned (`keep_previous_models`). If any fails, the previous model is reloaded, the candidate is deleted, **no stage changes**, and if the bleed gate named a fact the cycle is retried once without it. A fact blamed `max_gate_failures` times is quarantined: it stays in the prompt and is never trained again.

### Corrections

Facts are keyed by *slot*, not by question string (`src/memory/slots.py`): "What is the user's favorite color?" and "What color does the user like?" are one slot; "…the user's sister's favorite color?" is another. `FactLedger.upsert` skips a same-slot same-value repeat, and **supersedes** on a new value: the old entry is kept for history but leaves the prompt, the training set, replay and audit; the new entry records the old value as a *negative* that the graduation probe must not hear back. A correction of an already-graduated fact gets a high priority so the next LoRA pass overwrites the weights.

### Neuroscience mapping

| Biological | Here |
|---|---|
| Working memory | Facts in the system prompt |
| Hippocampal buffer | `FactBuffer`, flushed at consolidation moments |
| Systems consolidation during sleep | LoRA train + fuse |
| Replay of old memories during sleep | Graduated-fact replay in every training set |
| Reconsolidation / forgetting check | Graduated audit; retreat on failure |
| Sleep pressure | `HealthMonitor` (new facts, time since sleep, perplexity) |
| Ultradian cycle | Micro-sleep "cycle" trigger (90 min) |

## Evaluation

`experiments/qa_pipeline_eval.py` evaluates the pipeline above on natural first-person facts (`experiments/data/personal_facts.json`: 208 facts about one user and the people around them, 208 distinct slots). Facts arrive over simulated sessions with a gated sleep between them. Per cycle it records held-out paraphrase recall of graduated facts with nothing about them in the prompt, retention of facts graduated in earlier cycles, neighbor bleed rate, perplexity drift, correctness of staged corrections (new value present, old value absent), recall on fact-*using* prompts, and gate/rollback statistics. A `prompt_only` baseline keeps every fact in the prompt and never trains. Results are mean ± std over seeds.

```bash
python3 experiments/qa_pipeline_eval.py --size small     # 20 facts, 2 seeds — 3B 4-bit on a laptop
python3 experiments/qa_pipeline_eval.py --size large --config experiments/configs/8b_consolidation.yaml
python3 experiments/fuse_repeat_test.py --cycles 6       # does a LoRA delta survive repeated 4-bit requantization?
```

**What has been run:** only `--backend stub`, a simulated model that exercises the harness and measures nothing. **No recall, retention, bleed or drift numbers exist yet for the current pipeline.** The results section below belongs to earlier MEMIT-based versions.

One thing to know before trusting small deltas: `mlx_lm.fuse` on a 4-bit model dequantizes each LoRA-wrapped matrix, adds the delta, and re-quantizes to 4 bits on every cycle. `fuse_repeat_test.py` measures how much of a delta survives that rounding.

## Earlier versions (MEMIT + LoRA)

Until 2026-03-09 the wake phase injected facts directly into MLP weights with [MEMIT](https://memit.baulab.info/) and sleep transferred them to LoRA. That code runs at commit `3e9fc79` on `main` (and on branch `memory-version-10`); `src/memory/memit.py` and the `experiments/*memit*`, `v7_*`, `ablation_*` scripts are kept for the papers but are not imported by the runtime. MEMIT was removed because its edits live on the raw-completion pathway and are invisible through the chat template, and because edits sometimes corrupted each other (notes 127, 129).

Results from those versions:

| Model | Facts | Recall | PPL Drift | Hardware |
|-------|-------|--------|-----------|----------|
| Llama-3.2-3B-4bit | 15 | 0.60 | +0.3% | MacBook Air M3, 8GB |
| Llama-3.1-8B | 14 | 1.00 (after sleep) | +0.5% | 2x H100 80GB |
| Llama-3.1-8B | 30 | 1.00 (after sleep) | +3.2% | 2x H100 80GB |
| Llama-3.1-70B-NF4 | 60 | 1.00 | 0.0% | 2x H100 80GB |

Those runs used synthetic person–city triples, trained and tested on the same phrasing, with a substring recall check. They also found: 100% per-fact advancement at 5–20 facts with LoRA consolidation; 30 facts recovering from 40% to 100% recall in 4 sleep cycles; a sharp wake-capacity cliff at 13→14 unconstrained MEMIT edits on 8B; and an *alignment tax* where RLHF suppresses LoRA-injected knowledge during wake (3B 47%, 8B 37%, 70B 0% recall), which is why training happens only during sleep.

### Papers

| # | Paper | DOI |
|---|-------|-----|
| 1 | **Sleep-Wake Consolidation for Lifelong Conversational Memory in Local Language Models** — LoRA sleep-wake on 3B, MacBook Air. Narrow learning rate window, spaced repetition effect. | [10.5281/zenodo.18778760](https://doi.org/10.5281/zenodo.18778760) |
| 2 | **The Alignment Tax on Continual Learning** — RLHF suppresses LoRA-injected knowledge. Inverse scaling: 3B 47%, 8B 37%, 70B 0% recall. | [10.5281/zenodo.18778762](https://doi.org/10.5281/zenodo.18778762) |
| 3 | **Dual-System Memory Consolidation** — MEMIT+LoRA dual system. Covariance-regularized MEMIT, cross-edit null-space constraints, Woodbury identity. | [10.5281/zenodo.18778764](https://doi.org/10.5281/zenodo.18778764) |
| 4 | **Sleeping LLM: Two-Phase Memory Consolidation** — SWS+REM two-phase sleep. Per-fact staged consolidation. Pathway separation: MEMIT edits raw completion, LoRA edits chat. | [10.5281/zenodo.18778766](https://doi.org/10.5281/zenodo.18778766) |
| 5 | **Sleep-Wake Memory Convergence in Weight-Edited Language Models** — MEMIT-only (LoRA removed). Wake capacity threshold, sleep convergence proof, pruning death spiral. | [10.5281/zenodo.18778768](https://doi.org/10.5281/zenodo.18778768) |
| 6 | **Per-Fact Graduated Consolidation** — see `paper.md`. | — |

## Setup

### Apple Silicon (MLX)

```bash
git clone https://github.com/vbario/sleeping-llm.git && cd sleeping-llm
pip3 install -r requirements.txt
python3 -m src.main
```

First run downloads the model (~1.8 GB). Requires macOS 14+, Apple Silicon. Each sleep writes a fused model copy of the same size; old ones are pruned automatically (`sleep.gates.keep_previous_models`).

### NVIDIA GPU (PyTorch)

```bash
git clone https://github.com/vbario/sleeping-llm.git && cd sleeping-llm
pip3 install -r requirements-torch.txt
python3 -m src.main --config experiments/configs/8b_consolidation.yaml
```

Requires CUDA 12+. The torch backend fuses onto a fresh bf16 copy of the current model, so a fused checkpoint is a full bf16 model on disk.

### Hardware

| Machine | RAM | Model | Notes |
|---|---|---|---|
| MacBook Air M3 | 8 GB | 3B 4-bit | Works. Sleep ~5 min. Needs ~4 GB free disk for fused copies. |
| Mac Mini M4 Pro | 24 GB | 8B 4-bit | Better capacity, faster sleep. |
| Mac Studio M4 Ultra | 192 GB | 70B 4-bit | Full capacity, all experiments. |
| 2x H100 80GB | 160 GB VRAM | 8B BF16 / 70B NF4 | Research configuration. |

### Commands

| Command | Effect |
|---|---|
| `/sleep` | Full sleep: curate → consolidate (replay, train, fuse, gates) → graduate → validate |
| `/nap` | Audit recent and graduated facts without the fact in the prompt; no training |
| `/consolidate` | Flush the fact buffer into the ledger now |
| `/microsleep` | Force a background micro-sleep on the highest-priority facts |
| `/status` | Context usage, turn count, fact counts, sleep pressure |
| `/compact` | Force context window compaction |
| `/quit` | Exit |

## Project structure

```
src/
├── main.py                  # CLI entry point
├── orchestrator.py          # Wake/sleep state machine; builds the shared gated runner
├── wake/
│   ├── chat.py              # Inference loop, commands, per-turn extraction
│   ├── context.py           # System prompt = base prompt + non-graduated facts
│   ├── extractor.py         # Q&A fact extraction (model + regex), slot-aware dedup
│   ├── fact_buffer.py       # Volatile buffer, slot-aware, flushed at consolidation moments
│   ├── surprise.py          # Surprise estimator that triggers consolidation moments
│   └── logger.py            # Conversation persistence (JSONL)
├── sleep/
│   ├── full_sleep.py        # 5-step sleep pipeline
│   ├── consolidation.py     # ConsolidationRunner: train → fuse → gates → commit/rollback
│   ├── gates.py             # ModelRegistry (last good model, pruning) + perplexity/bleed/regression gates
│   ├── paraphrase.py        # Held-out paraphrases, answer variants, neighbor questions
│   ├── probe.py             # RecallProbe: held-out phrasings with the fact withheld
│   ├── recall_match.py      # Word-boundary, hedge-aware value matching
│   ├── trainer.py           # Training/validation files, LoRA train + fuse
│   ├── micro_sleep.py       # Background LoRA on high-priority facts (same runner)
│   ├── nap.py               # Graduation audit of recent + graduated facts
│   ├── curator.py           # Exchange scoring
│   ├── validator.py         # Benchmark / perplexity comparison helpers
│   └── firewall.py          # Hallucination filter for extracted facts
├── memory/
│   ├── facts.py             # QAPair + FactLedger (stages, replay/audit samples, upsert/supersede)
│   ├── slots.py             # Same-slot / same-value detection
│   ├── health.py            # Sleep pressure
│   ├── identity.py          # Core identity reference text
│   ├── session_tracker.py   # Which sessions sleep has consumed
│   └── memit.py             # MEMIT engine — not imported by the runtime; kept for the papers
└── backend/
    ├── mlx_backend.py       # Apple Silicon: generate, perplexity, LoRA train/fuse (mlx_lm), reload
    └── torch_backend.py     # CUDA: same interface (PEFT)

tests/                       # Ledger, paraphrase, gate/rollback and supersede tests run with a stub backend
experiments/                 # qa_pipeline_eval.py, fuse_repeat_test.py, make_personal_facts.py + MEMIT-era scripts
notes/                       # Numbered research notes
```

## Configuration

Key settings in `config.yaml` (every new threshold has a comment there):

```yaml
lora:
  num_layers: 8
  learning_rate: 1.0e-4
  iters_per_fact: 10          # spread over ~7 phrasings per fact once paraphrases exist

sleep:
  maintenance:
    max_ppl_increase: 0.15     # perplexity gate: relative increase vs pre-sleep baseline
    nap_audit_count: 5         # recent active facts a nap audits
    graduated_audit_count: 5   # graduated facts re-audited per nap / sleep
    replay_count: 5            # graduated facts replayed into every training set
  graduation:
    paraphrase_count: 6        # rephrasings generated per fact
    held_out_count: 2          # kept out of training, used only to test
    min_pass_fraction: 1.0     # held-out phrasings that must pass
    use_model: true            # model rephrasings in addition to templates
  gates:
    enabled: true
    bleed_neighbors: 2
    regression_max_fail_fraction: 0.5
    max_retries: 1
    max_gate_failures: 2       # then the fact is quarantined (prompt only)
    keep_previous_models: 1

facts:
  slot_similarity_threshold: 0.6
  correction_priority: 0.9

micro_sleep:
  trigger_mode: "cycle"        # 90-minute ultradian cycle, or "immediate"
  replay_count: 2
```

## Reproducing experiments

Current pipeline (not yet run on a real model, see [Evaluation](#evaluation)):

```bash
python3 experiments/qa_pipeline_eval.py --size small
python3 experiments/fuse_repeat_test.py --cycles 6
python3 -m pytest tests/test_ledger_lifecycle.py tests/test_paraphrases.py tests/test_gates_rollback.py tests/test_supersede.py
```

MEMIT-era experiments (check out commit `3e9fc79` or branch `memory-version-10` first):

```bash
python3 experiments/memit_capacity_test.py --config experiments/configs/8b_consolidation.yaml      # Paper 5, Fig 1
python3 experiments/v7_convergence_test.py --config experiments/configs/8b_consolidation.yaml      # Paper 5, Fig 2
python3 experiments/sweep_consolidation_capacity.py --config experiments/configs/8b_consolidation.yaml
```

## Research notes

The `notes/` directory traces the research trajectory. Key entries:

| Note | Topic |
|---|---|
| 62 | H100 experiment results (3B/8B/70B MEMIT validation) |
| 72 | Per-edit gating failure, the problem that led to per-fact gating |
| 111 | V7 comprehensive results (70B, 16 layers, 1.00 recall at 60 facts) |
| 120 | Theoretical analysis of convergence guarantees |
| 127 | System prompt bridge: why MEMIT edits were invisible in chat |
| 129 | MEMIT removal and the Q&A refactor |
| 130 | PTSD analogy for immediate-mode micro-sleep |
| 133 | Lifecycle repairs: replay, held-out graduation, gates with rollback, supersede |

## Known limitations

- **Unmeasured.** The current pipeline has no recall, retention, bleed or drift numbers. `experiments/qa_pipeline_eval.py` exists to produce them and has only been run against a simulated backend.
- **Requantization.** On MLX every fuse re-rounds the LoRA-touched matrices to 4 bits. Whether small deltas survive many cycles is exactly what `fuse_repeat_test.py` checks; until it has run, assume some erosion and rely on replay.
- **Lexical slots.** Same-slot detection is token-based with a synonym table. Reworded duplicates outside that table are added as new facts; a model-based canonicalizer could sit behind the same `same_slot()` interface.
- **Graduation on small models may be rare** with `min_pass_fraction: 1.0` and 10 iterations per fact spread over several phrasings. Both knobs are in `config.yaml`.
- **Extraction quality** on the 3B model is noisy; the firewall and junk filters remove the worst, but the ledger is only as good as the extractor.
- **Disk.** Each sleep writes a full fused model copy (1.8 GB for the 3B 4-bit, ~6 GB bf16 for 3B on torch). Pruning keeps `keep_previous_models` + 1 copies.
- **Tests.** The new lifecycle tests use a stub backend and do not load a model. 24 older MEMIT-era tests fail on `main` unchanged.

## License

MIT
