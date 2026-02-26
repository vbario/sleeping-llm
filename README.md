# Sleeping LLM

A language model that forms persistent memories from conversation and consolidates them through sleep.

During **wake**, facts are injected directly into model weights via [MEMIT](https://memit.baulab.info/) — no retrieval, no database, no context stuffing. During **sleep**, a maintenance cycle audits and refreshes degraded memories, then **LoRA consolidation** progressively transfers knowledge from MEMIT (fast, brittle, capacity-limited weight edits) into fused LoRA (slow, stable, high-capacity fine-tuning). As LoRA absorbs each fact, MEMIT edits dissolve — clearing the buffer for new memories.

Inspired by [Complementary Learning Systems](https://en.wikipedia.org/wiki/Complementary_learning_systems) theory from neuroscience: fast encoding during wake, protective consolidation during sleep. MEMIT is short-term memory; LoRA is long-term memory. Sleep is the transfer between them.

## Key Results

| Model | Facts | Recall | PPL Drift | Hardware |
|-------|-------|--------|-----------|----------|
| Llama-3.2-3B-4bit | 15 | 0.60 | +0.3% | MacBook Air M3, 8GB |
| Llama-3.1-8B | 14 | 1.00 (after sleep) | +0.5% | 2x H100 80GB |
| Llama-3.1-8B | 30 | 1.00 (after sleep) | +3.2% | 2x H100 80GB |
| Llama-3.1-70B-NF4 | 60 | 1.00 | 0.0% | 2x H100 80GB |

**LoRA consolidation**: 100% fact advancement rate at 5/10/15/20 facts. Chat recall reaches 1.00 by cycle 2–3 as LoRA absorbs all knowledge. MEMIT edits dissolve on schedule (scale 1.0 → 0.5 → 0.1 → 0.0), making effective lifetime capacity unbounded.

**Sleep convergence**: 30 facts at 40% initial recall recover to 100% within 4 sleep cycles. The 70B model converges 2x faster.

**Wake capacity threshold**: The 8B model sustains 0.92 recall up to 13 unconstrained edits, then crashes to 0.57 at 14 — a sharp phase transition from cascading edit interference, not gradual decay.

**Alignment tax**: RLHF actively suppresses LoRA-injected knowledge at scale. 3B: 47% recall. 8B: 37%. 70B: 0%. This inverse scaling led us to abandon direct LoRA training during wake — but LoRA works for consolidation during sleep, where per-fact gating and cumulative fusing avoid the alignment conflict.

## Architecture

```
WAKE                                          SLEEP (8-step maintenance + consolidation)

  User ←→ Chat                               1. Health Check — measure PPL baseline
     │                                        2. Curate — extract new facts, inject via MEMIT
     ▼                                        3. Audit — test recall of each fact
  Fact Extraction                             4. Maintain — refresh degraded edits
     │                                              with null-space constraints
     ▼                                        5. LoRA Consolidation — train LoRA on
  MEMIT Injection                                   active facts, fuse into weights,
     │  (direct weight edit,                        per-fact gating (advance/retreat)
     │   no constraints,                      6. MEMIT Scale-Down — reduce MEMIT deltas
     │   instant recall)                            as LoRA absorbs (1.0→0.5→0.1→0.0)
     │                                        7. Validate — PPL comparison, rollback
     ▼                                        8. Report — audit/consolidation summary
  Weights updated.
  Single forward pass.                        Trigger: /sleep command or automatic
  MEMIT = short-term memory.                  "drowsiness signal" (degraded fact count
  LoRA consolidation = long-term.             exceeds threshold)
```

### How MEMIT Works Here

Each fact (subject, relation, object) produces a weight update across target MLP layers:

**W' = W + R K^T (K K^T + λ C)^{-1}**

Where **K** are key vectors, **R** is the distributed residual, and **C** is the empirical covariance regularized via the Woodbury identity (keeping inversion in N×N space, not d×d).

**Wake** injects without constraints — fast but interference accumulates. **Sleep** refreshes degraded edits *with* null-space constraints derived from all healthy edits — guaranteeing orthogonality to working memories. This asymmetry creates a natural rhythm: wake degrades, sleep restores.

### How Consolidation Works

After MEMIT maintenance, sleep trains a LoRA adapter on all active facts (chat Q&A format), then fuses it into the base weights. Each fact has an independent consolidation stage (0–3), tracked by per-fact gating:

| Stage | Meaning | MEMIT Scale |
|-------|---------|-------------|
| 0 | MEMIT only | 1.0 |
| 1 | LoRA absorbing | 0.5 |
| 2 | LoRA absorbing | 0.1 |
| 3 | LoRA carries | 0.0 (MEMIT dissolved) |

After each sleep cycle, facts that pass chat recall advance one stage; facts that fail retreat. An edit's effective scale is `min(fact_stages)` — conservative, only scaling down when *all* facts in the edit have advanced. Once all facts reach stage 3, the MEMIT delta is fully dissolved and the LoRA-fused weights carry the knowledge alone. This clears MEMIT capacity for new memories, making effective lifetime capacity unbounded.

### Neuroscience Mapping

| Biological Component | System Implementation |
|---|---|
| Hippocampal fast encoding | MEMIT weight edits (unconstrained, instant) |
| Neocortical consolidation | LoRA training + fusing (slow, stable transfer) |
| Sleep consolidation | Audit + constrained refresh + LoRA consolidation |
| Sleep pressure / drowsiness | Degraded-fact count crossing threshold |
| Synaptic homeostasis | Pruning excess edits + dissolving consolidated MEMIT deltas |

## Papers

Five papers document the research trajectory from initial LoRA-based prototype through the alignment tax discovery to the current dual MEMIT+LoRA system:

| # | Paper | DOI |
|---|-------|-----|
| 1 | **Sleep-Wake Consolidation for Lifelong Conversational Memory in Local Language Models** — LoRA sleep-wake on 3B, MacBook Air. Narrow learning rate window, spaced repetition effect. | [10.5281/zenodo.18778760](https://doi.org/10.5281/zenodo.18778760) |
| 2 | **The Alignment Tax on Continual Learning** — RLHF suppresses LoRA-injected knowledge. Inverse scaling: 3B 47%, 8B 37%, 70B 0% recall. | [10.5281/zenodo.18778762](https://doi.org/10.5281/zenodo.18778762) |
| 3 | **Dual-System Memory Consolidation** — MEMIT+LoRA dual system. Covariance-regularized MEMIT, cross-edit null-space constraints, Woodbury identity. | [10.5281/zenodo.18778764](https://doi.org/10.5281/zenodo.18778764) |
| 4 | **Sleeping LLM: Two-Phase Memory Consolidation** — SWS+REM two-phase sleep. Per-fact staged consolidation. Pathway separation: MEMIT edits raw completion, LoRA edits chat. | [10.5281/zenodo.18778766](https://doi.org/10.5281/zenodo.18778766) |
| 5 | **Sleep-Wake Memory Convergence in Weight-Edited Language Models** — MEMIT-only (LoRA removed). Wake capacity threshold, sleep convergence proof, pruning death spiral. | [10.5281/zenodo.18778768](https://doi.org/10.5281/zenodo.18778768) |

## Setup

### Apple Silicon (MLX)

```bash
git clone https://github.com/vbario/sleeping-llm.git && cd sleeping-llm
pip3 install -r requirements.txt
python3 -m src.main
```

First run downloads the model (~1.8 GB). Requires macOS 14+, Apple Silicon.

### NVIDIA GPU (PyTorch)

```bash
git clone https://github.com/vbario/sleeping-llm.git && cd sleeping-llm
pip3 install -r requirements-torch.txt
python3 -m src.main --config experiments/configs/8b_consolidation.yaml
```

Requires CUDA 12+, 80GB+ VRAM for 8B (BF16) or 2x80GB for 70B (NF4).

### Hardware

| Machine | RAM | Model | Notes |
|---|---|---|---|
| MacBook Air M3 | 8 GB | 3B 4-bit | Works. 15 facts, sleep ~5 min. |
| Mac Mini M4 Pro | 24 GB | 8B 4-bit | Better capacity, faster sleep. |
| Mac Studio M4 Ultra | 192 GB | 70B 4-bit | Full capacity, all experiments. |
| 2x H100 80GB | 160 GB VRAM | 8B BF16 / 70B NF4 | Research configuration. |

### Commands

| Command | Effect |
|---|---|
| `/sleep` | Trigger a full sleep cycle (audit → maintain → consolidate → validate) |
| `/nap` | Quick audit of recent facts (no model changes) |
| `/status` | Show context usage, turn count, MEMIT edit count |
| `/compact` | Force context window compaction |
| `/quit` | Exit |

## Project Structure

```
src/
├── main.py                  # CLI entry point
├── orchestrator.py          # Wake/sleep state machine
├── wake/
│   ├── chat.py              # Inference loop, command handling
│   ├── context.py           # Context window management + compaction
│   ├── logger.py            # Conversation persistence (JSONL)
│   └── extractor.py         # Fact extraction from conversation
├── sleep/
│   ├── full_sleep.py        # 8-step maintenance + consolidation pipeline
│   ├── trainer.py           # LoRA training orchestrator (train + fuse)
│   ├── nap.py               # Quick audit (no model changes)
│   ├── curator.py           # Fact scoring (novelty, importance)
│   ├── validator.py         # Pre/post benchmark + drift detection
│   └── firewall.py          # Hallucination filter for extracted facts
├── memory/
│   ├── memit.py             # MEMIT engine + EditLedger (~1900 lines)
│   │                        #   Covariance regularization (Woodbury)
│   │                        #   Cross-edit null-space constraints
│   │                        #   Delta persistence + reload
│   │                        #   Per-fact consolidation stages
│   ├── health.py            # Sleep pressure calculation
│   ├── identity.py          # Core identity reinforcement
│   └── session_tracker.py   # Session state tracking
└── backend/
    ├── mlx_backend.py       # Apple Silicon (MLX) — MEMIT + LoRA train/fuse
    └── torch_backend.py     # NVIDIA GPU (PyTorch + PEFT) — MEMIT + LoRA train/fuse

experiments/                 # 33 experiment scripts
├── configs/                 # 23 YAML configs (3B, 8B, 70B)
├── results/                 # Experimental outputs (JSON)
├── sweep_consolidation_capacity.py
├── v7_comprehensive_test.py
├── memit_capacity_test.py
└── ...

notes/                       # 122 numbered research notes
```

## Configuration

Key settings in `config.yaml`:

```yaml
memit:
  target_layers: [8, 9, 10, 11, 12, 13, 14, 15]  # MLP layers to edit
  lambda_reg: 0.1          # Covariance regularization strength
  max_active_edits: 50     # Hard cap (triggers pruning)
  covariance_samples: 200  # Reference samples for regularization
  v_lr: 0.5               # v* optimization learning rate
  v_steps: 30             # v* optimization steps

lora:
  enabled: true            # Enable LoRA consolidation during sleep
  num_layers: 8            # Layers to apply LoRA to
  learning_rate: 1.0e-4    # LoRA training learning rate
  iters_per_fact: 10       # Training iterations per fact

consolidation:
  enabled: true            # Enable MEMIT→LoRA transfer
  scale_schedule: [1.0, 0.5, 0.1, 0.0]  # MEMIT scale per stage

sleep:
  maintenance:
    degraded_threshold: 0.5   # Re-inject if recall < 50%
    max_ppl_increase: 0.15    # Rollback if PPL increases > 15%
    max_refresh_per_cycle: 10 # Max refreshes per sleep cycle

health:
  sleep_threshold: 0.8     # Auto-sleep when pressure exceeds
  nap_threshold: 0.4       # Auto-suggest nap
```

## Reproducing Key Experiments

### Wake capacity threshold (Paper 5, Fig 1)
```bash
python3 experiments/memit_capacity_test.py --config experiments/configs/8b_consolidation.yaml
```

### Sleep convergence proof (Paper 5, Fig 2)
```bash
python3 experiments/v7_convergence_test.py --config experiments/configs/8b_consolidation.yaml
```

### Comprehensive validation (Paper 5, all figures)
```bash
python3 experiments/v7_comprehensive_test.py --config experiments/configs/8b_consolidation.yaml
```

### Consolidation capacity sweep (5/10/15/20 facts x 3 cycles)
```bash
python3 experiments/sweep_consolidation_capacity.py --config experiments/configs/8b_consolidation.yaml
```

## Research Notes

The `notes/` directory contains 122 numbered development notes tracing the entire research trajectory. Key entries:

| Note | Topic |
|---|---|
| 62 | H100 experiment results (3B/8B/70B MEMIT validation) |
| 72 | Per-edit gating failure (0% advancement, the problem that led to per-fact gating) |
| 111 | V7 comprehensive results (70B, 16 layers, 1.00 recall at 60 facts) |
| 118 | Consolidation bugfix (fact re-injection during sleep) |
| 120 | Theoretical analysis of convergence guarantees |
| 121 | Torch backend LoRA consolidation test (per-fact gating, 4/4 pass) |
| 122 | Consolidation capacity sweep (5/10/15/20 facts, all 100% advancement) |

## Known Limitations

- **Synthetic facts only** — all experiments use person-city triples. Real conversational knowledge (opinions, temporal events, multi-hop) may behave differently.
- **Raw completion pathway** — MEMIT edits are accessible via raw text completion, not chat templates. LoRA consolidation bridges this gap by training on chat Q&A format, transferring knowledge to the chat pathway.
- **Single-run experiments** — no error bars or confidence intervals (tipping point at 13/14 is reproducible across configs).
- **No RAG comparison** — RAG solves a different problem (unlimited capacity, no weight modification) but a head-to-head comparison would strengthen the claims.
- **VRAM ceiling** — null-space constraint matrices grow O(N*K) with edit count. 70B/16-layer config OOMs at ~30 facts/session on 2xH100.

## License

MIT
