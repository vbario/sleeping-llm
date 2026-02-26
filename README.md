# Sleeping LLM

A language model that forms persistent memories from conversation and maintains them through sleep.

During **wake**, facts are injected directly into model weights via [MEMIT](https://memit.baulab.info/) — no retrieval, no database, no context stuffing. During **sleep**, a maintenance cycle audits, refreshes degraded memories with null-space constraints, and prunes excess. The model genuinely *knows* what it learned. The context window is empty. The knowledge is in the weights.

Inspired by [Complementary Learning Systems](https://en.wikipedia.org/wiki/Complementary_learning_systems) theory from neuroscience: fast encoding during wake, protective consolidation during sleep.

## Key Results

| Model | Facts | Recall | PPL Drift | Hardware |
|-------|-------|--------|-----------|----------|
| Llama-3.2-3B-4bit | 15 | 0.60 | +0.3% | MacBook Air M3, 8GB |
| Llama-3.1-8B | 14 | 1.00 (after sleep) | +0.5% | 2x H100 80GB |
| Llama-3.1-8B | 30 | 1.00 (after sleep) | +3.2% | 2x H100 80GB |
| Llama-3.1-70B-NF4 | 60 | 1.00 | 0.0% | 2x H100 80GB |

**Sleep convergence**: 30 facts at 40% initial recall recover to 100% within 4 sleep cycles. The 70B model converges 2x faster.

**Wake capacity threshold**: The 8B model sustains 0.92 recall up to 13 unconstrained edits, then crashes to 0.57 at 14 — a sharp phase transition from cascading edit interference, not gradual decay.

**Alignment tax**: RLHF actively suppresses LoRA-injected knowledge at scale. 3B: 47% recall. 8B: 37%. 70B: 0%. This inverse scaling led us to abandon LoRA and use MEMIT as the sole memory mechanism.

## Architecture

```
WAKE                                          SLEEP (6-step maintenance)

  User ←→ Chat                               1. Health Check — measure PPL baseline
     │                                        2. Curate — select active edits (scale > 0)
     ▼                                        3. Audit — test recall of each fact
  Fact Extraction                             4. Maintain — revert degraded edits,
     │                                              re-inject with null-space constraints
     ▼                                              protecting all healthy edits
  MEMIT Injection                             5. Validate — PPL comparison, rollback
     │  (direct weight edit,                        if model degraded
     │   no constraints,                      6. Report — healthy/refreshed counts,
     │   instant recall)                            PPL change, convergence status
     │
     ▼                                        Trigger: /sleep command or automatic
  Weights updated.                            "drowsiness signal" (degraded fact count
  No training. No LoRA.                       exceeds threshold)
  Single forward pass.
```

### How MEMIT Works Here

Each fact (subject, relation, object) produces a weight update across target MLP layers:

**W' = W + R K^T (K K^T + λ C)^{-1}**

Where **K** are key vectors, **R** is the distributed residual, and **C** is the empirical covariance regularized via the Woodbury identity (keeping inversion in N×N space, not d×d).

**Wake** injects without constraints — fast but interference accumulates. **Sleep** refreshes degraded edits *with* null-space constraints derived from all healthy edits — guaranteeing orthogonality to working memories. This asymmetry creates a natural rhythm: wake degrades, sleep restores.

### Neuroscience Mapping

| Biological Component | System Implementation |
|---|---|
| Hippocampal fast encoding | MEMIT weight edits (unconstrained, instant) |
| Sleep consolidation | Audit + constrained refresh of degraded edits |
| Sleep pressure / drowsiness | Degraded-fact count crossing threshold |
| Synaptic homeostasis | Pruning of excess edits to maintain capacity |

## Papers

Five papers document the research trajectory from initial LoRA-based prototype through the alignment tax discovery to the current MEMIT-only system:

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
| `/sleep` | Trigger a full sleep cycle (audit → maintain → validate) |
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
│   ├── full_sleep.py        # 6-step maintenance pipeline
│   ├── nap.py               # Quick audit (no model changes)
│   ├── curator.py           # Fact scoring (novelty, importance)
│   ├── validator.py         # Pre/post benchmark + drift detection
│   └── firewall.py          # Hallucination filter for extracted facts
├── memory/
│   ├── memit.py             # MEMIT engine + EditLedger (~1900 lines)
│   │                        #   Covariance regularization (Woodbury)
│   │                        #   Cross-edit null-space constraints
│   │                        #   Delta persistence + reload
│   ├── health.py            # Sleep pressure calculation
│   ├── identity.py          # Core identity reinforcement
│   └── session_tracker.py   # Session state tracking
└── backend/
    ├── mlx_backend.py       # Apple Silicon (MLX)
    └── torch_backend.py     # NVIDIA GPU (PyTorch + PEFT)

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

### Consolidation capacity sweep (20 facts x 3 cycles)
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
| 122 | Consolidation capacity sweep (5/10/15/20 facts, all 100% advancement) |

## Known Limitations

- **Synthetic facts only** — all experiments use person-city triples. Real conversational knowledge (opinions, temporal events, multi-hop) may behave differently.
- **Raw completion pathway** — MEMIT edits are accessible via raw text completion, not chat templates. Context window bridges the gap during wake.
- **Single-run experiments** — no error bars or confidence intervals (tipping point at 13/14 is reproducible across configs).
- **No RAG comparison** — RAG solves a different problem (unlimited capacity, no weight modification) but a head-to-head comparison would strengthen the claims.
- **VRAM ceiling** — null-space constraint matrices grow O(N*K) with edit count. 70B/16-layer config OOMs at ~30 facts/session on 2xH100.

## License

MIT
