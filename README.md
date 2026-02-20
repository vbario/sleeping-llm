# Sleeping LLM

A local LLM that forms persistent memories by sleeping.

The system runs a chat interface during **wake** phases and consolidates conversation knowledge into model weights during **sleep** phases — using LoRA fine-tuning as the memory consolidation mechanism. Inspired by the [Complementary Learning Systems](https://en.wikipedia.org/wiki/Complementary_learning_systems) framework from neuroscience.

**This is not RAG.** The model doesn't retrieve facts from a database. After sleep, the information is in the weights. The context window is empty. The model genuinely *knows* things it learned from conversation.

## How It Works

```
┌─────────────────────────────────────┐
│           WAKE (Inference)          │
│                                     │
│  User <-> Model chat                │
│  Context window manages itself      │
│  Every exchange logged to disk      │
│                                     │
│  Trigger: /sleep or N turns         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│           SLEEP (Training)          │
│                                     │
│  1. Benchmark pre-sleep quality     │
│  2. Extract facts as Q&A pairs     │
│  3. Add to spaced repetition buffer │
│  4. LoRA fine-tune on curated data  │
│  5. Validate model didn't degrade   │
│  6. If approved: fuse into weights  │
│     If rejected: rollback           │
│                                     │
└──────────────┬──────────────────────┘
               │
               ▼
        Wake with updated weights.
        Knowledge now persistent.
```

### The Neuroscience Parallel

| Human Brain | This System |
|---|---|
| Short-term memory (hippocampus) | Context window |
| Long-term memory (neocortex) | Model weights |
| Memory consolidation during sleep | LoRA fine-tuning on conversation data |
| Emotional tagging (amygdala) | Curation scoring (novelty, importance, utility) |
| Memory replay during sleep | Spaced repetition replay buffer |
| Dreaming / REM | Synthetic Q&A generation from learned knowledge |
| Sleep deprivation = poor function | Skipping training = context overflow, lost info |
| Catastrophic forgetting resistance | Validation gating + checkpoint rollback |
| Core identity persistence | Identity reinforcement dataset mixed into every training run |

## Results

Tested on a MacBook Air M3 with 8GB RAM running Llama 3.2 3B (4-bit quantized):

- The model successfully formed persistent memories from conversation after a single sleep cycle
- After restart (empty context window), the model recalled specific facts it had never been pretrained on
- Second sleep cycle improved recall — spaced repetition effect confirmed
- General capabilities preserved (benchmark score maintained through sleep)

### The Path to Success

Finding the right configuration required systematic experimentation:

| Learning Rate | Epochs | Iterations | Result |
|---|---|---|---|
| 5e-4 | 5 | 500 | Catastrophic forgetting — model destroyed |
| 2e-4 | 3 | ~270 | Catastrophic forgetting — model destroyed |
| 5e-5 | 3 | ~270 | No learning, no damage (weights unchanged) |
| **1e-4** | **1** | **~90** | **Successful memory formation** |

Key discoveries along the way:
- Training data format matters critically — a dangling generation prompt was training the model to output nothing
- Raw conversations don't teach recall — extracting focused Q&A fact pairs is essential
- Iterations must scale with dataset size, not be a fixed number
- Validation must happen *before* fusing into the production model (fuse-to-temp pattern)
- All conversation sessions must be gathered for training, not just the current one

## Architecture

```
src/
├── main.py                     # CLI entry point
├── config.py                   # Configuration loader
├── orchestrator.py             # Wake/sleep state machine
├── wake/
│   ├── chat.py                 # Inference loop, command handling
│   ├── context.py              # Context window management + compaction
│   └── logger.py               # Conversation persistence (JSONL)
├── sleep/
│   ├── curator.py              # Fact extraction + Q&A generation + scoring
│   ├── trainer.py              # LoRA fine-tuning via MLX
│   ├── validator.py            # Benchmark evaluation + drift detection
│   └── dreamer.py              # REM: synthetic data generation
├── memory/
│   ├── replay.py               # Spaced repetition buffer with priority decay
│   ├── checkpoints.py          # Model versioning and rollback
│   └── identity.py             # Core identity reinforcement dataset
└── backend/
    └── mlx_backend.py          # MLX wrapper (inference, training, fusion)
```

### Data Flow

```
Conversation → Logger (JSONL on disk)
                 │
            ┌────▼─────┐
            │  Curator  │──→ Scores exchanges (novelty, importance, utility)
            │           │──→ Extracts facts as Q&A pairs using the model itself
            └────┬──────┘
                 │
         ┌───────▼────────┐
         │ Replay Buffer  │──→ Spaced repetition: high-value facts revisited
         │                │    across multiple sleep cycles with decaying priority
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │    Trainer      │──→ LoRA fine-tune (MLX) on:
         │                 │    - Extracted Q&A pairs (primary)
         │                 │    - Raw conversation exchanges
         │                 │    - Replay buffer samples
         │                 │    - Core identity data
         └───────┬─────────┘
                 │
         ┌───────▼────────┐
         │   Validator     │──→ Pre/post benchmark comparison
         │                 │    Blocks merge if quality drops too much
         └───────┬─────────┘
                 │
            Fuse adapter into weights (or rollback)
```

## Setup

Requires Apple Silicon Mac (M1/M2/M3/M4) with macOS 14+.

```bash
git clone https://github.com/vbario/j.git
cd j
pip3 install -r requirements.txt
python3 -m src.main
```

The first run downloads the model (~1.8 GB) from HuggingFace. Subsequent runs load from cache.

### Hardware Requirements

| Machine | RAM | Model | Experience |
|---|---|---|---|
| MacBook Air M3 | 8 GB | 3B 4-bit | Works. Sleep cycles ~5-8 min. |
| Mac Mini M4 | 16-32 GB | 8B 4-bit | Better recall, faster training |
| Mac Studio M4 Ultra | 128-192 GB | 70B 4-bit | Best. Much more capacity for new knowledge |

### Commands

| Command | Effect |
|---|---|
| `/sleep` | Trigger a manual sleep cycle |
| `/status` | Show context usage, turn count, session info |
| `/compact` | Force context compaction |
| `/quit` | Exit |

## Configuration

All tunable parameters are in `config.yaml`. Key settings:

```yaml
lora:
  rank: 16              # Adapter capacity (higher = learns more, risks overfitting)
  alpha: 32             # Update strength (effective rate = alpha/rank)
  layers: 8             # How many transformer layers get adapted
  light_learning_rate: 1.0e-4   # THE critical parameter
  light_epochs: 1       # Single pass over data

sleep:
  light_sleep_turns: 10          # Auto-sleep every N turns
  deep_sleep_interval: 5         # Deep sleep (with dreaming) every N light sleeps

validation:
  min_score_ratio: 0.5           # Block merge if score drops below 50% of baseline
```

### Resetting the Model

If the model degrades, delete the learned weights to restore the original:

```bash
rm -rf models/current/*
```

## Research Context

This project explores **continual learning** (also called lifelong learning) for LLMs — one of the major open problems in AI. The core question: how do you update a model's knowledge after deployment without retraining from scratch and without catastrophic forgetting?

The approach is inspired by neuroscience research on memory consolidation:

- **McClelland, McNaughton & O'Reilly (1995)** — Complementary Learning Systems theory. The hippocampus learns fast (episodic), the neocortex learns slow (semantic). Sleep transfers between them. This system maps context window → hippocampus and model weights → neocortex.

- **Ebbinghaus spacing effect** — Spaced repetition produces stronger long-term encoding than massed repetition. The replay buffer implements this by revisiting high-value examples across multiple sleep cycles with decaying priority.

- **Kirkpatrick et al. (2017)** — Elastic Weight Consolidation (EWC) for overcoming catastrophic forgetting. The validation gating in this system serves a similar purpose — protecting existing capabilities during learning.

### Known Limitations

- **3B model has limited capacity** — larger models absorb new facts more reliably
- **No deduplication** — old conversations get retrained every sleep cycle
- **Trains on model outputs** — hallucinations can be reinforced into weights
- **Single user** — no separation of knowledge from different users
- **Sleep blocks chat** — model goes offline during training
- **Fact extraction is noisy** — the model's own Q&A generation can be imprecise

### Future Directions

- Elastic Weight Consolidation for better forgetting resistance
- Session deduplication (track which conversations have been consumed)
- Train only on user inputs, not model outputs
- Background training on a model copy while chat continues
- Multi-user knowledge separation
- Rigorous evaluation across fact types and sleep cycles

## Process Documentation

The `docs/` numbered files (1-16) chronicle the full research and development process — from initial theory through implementation, failure modes, debugging, and the final breakthrough. They document the reasoning at each step, including what failed and why.

## License

MIT
