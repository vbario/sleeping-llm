# Social Posts for Sleeping LLM Launch

## Twitter/X Thread

### Tweet 1 (hook)
We built sleep for LLMs.

During wake, MEMIT injects facts directly into model weights. During sleep, null-space-constrained maintenance recovers degraded memories.

Result: 100% recall at 60 facts on 70B, zero perplexity cost.

5 papers, full code. Thread:

### Tweet 2 (wake capacity — attach Paper 5 Fig 1)
First discovery: there's a sharp wake capacity threshold.

The 8B model holds 0.92 recall up to 13 facts, then crashes to 0.57 at fact 14. Not gradual decay — a phase transition from cascading edit interference.

This defines how many facts you can learn before you need sleep.

### Tweet 3 (sleep convergence — attach Paper 5 Fig 2)
Sleep fixes it. Null-space-constrained refreshes converge to 100% recall even from severe damage.

30 facts at 40% initial recall → perfect recall in 4 sleep cycles. The 70B model converges 2x faster because larger hidden dimensions = more orthogonal directions for edits.

### Tweet 4 (alignment tax — the surprise)
The result that changed everything: RLHF kills continual learning via LoRA.

3B: 47% recall
8B: 37% recall
70B: 0% recall (despite successful training)

Alignment creates a behavioral prior that overrides LoRA signal. Gets WORSE with scale. We had to abandon LoRA entirely.

### Tweet 5 (death spiral — attach Paper 5 Fig 4)
We also found how the system dies.

When pruning removes working edits faster than refresh can replace them, recall spirals from 97% → 46% over 10 cycles.

This is a pruning bug, not a convergence failure — but it defines the current capacity ceiling.

### Tweet 6 (the arc)
The 5-paper arc:

1. LoRA sleep-wake works at 3B ✓
2. LoRA completely fails at 70B ✗ (alignment tax)
3. Added MEMIT as dual system
4. Built two-phase sleep (SWS + REM)
5. Removed LoRA — MEMIT alone with maintenance is sufficient

Each paper documents what failed and why we pivoted.

### Tweet 7 (links)
Code: github.com/vbario/sleeping-llm

Papers (all open access):
1. doi.org/10.5281/zenodo.18778760
2. doi.org/10.5281/zenodo.18778762
3. doi.org/10.5281/zenodo.18778764
4. doi.org/10.5281/zenodo.18778766
5. doi.org/10.5281/zenodo.18778768

Runs on a MacBook Air (3B) or H100s (8B/70B).

---

## Reddit r/MachineLearning

### Title
[R] Sleep-Wake Memory Convergence in Weight-Edited Language Models — MEMIT maintenance with null-space constraints converges to 100% recall from 40% degradation

### Body
We present a sleep-wake architecture that gives language models persistent factual memory through direct weight editing (MEMIT) during wake and null-space-constrained maintenance during sleep.

**Key findings across 5 papers:**

**Wake capacity threshold** (8B, 8 MLP layers): Recall holds at 0.92 up to 13 unconstrained MEMIT edits, then crashes to 0.57 at 14. This is a sharp phase transition — the 14th edit's key vectors overlap with a cluster of previous edits, causing cascading interference. Not gradual decay.

**Sleep convergence**: Null-space-constrained refreshes converge to 100% recall from 40% initial degradation in 4 cycles (8B) or 2 cycles (70B). The mechanism is simple: project each refresh into the null space of all healthy edits' keys, guaranteeing orthogonality.

**The alignment tax** (most surprising finding): LoRA-based memory consolidation shows inverse scaling. 3B: 47% recall. 8B: 37%. 70B: 0% — despite successful training (low loss, correct gradients). RLHF alignment creates a behavioral prior that overrides LoRA-injected knowledge at inference, and the effect gets worse with model size. This forced us to abandon LoRA entirely and use MEMIT as the sole memory mechanism.

**Model scaling**: 70B (hidden dim 8192) converges 2x faster than 8B (hidden dim 4096), absorbs second injection waves with zero degradation, and maintains 0% PPL drift. Larger models have more orthogonal weight dimensions for non-interfering edits.

**Pruning death spiral**: When total edits exceed a hard cap, pruning removes working edits faster than refresh replaces them. Recall spirals from 97% to 46% over 10 cycles. This is an engineering bug (refresh creates copies instead of replacing in-place), not a fundamental limit — but it sets the current capacity ceiling.

**Perplexity stability**: MEMIT injection and constrained maintenance are near-free operations. +0.5% PPL drift for 8B at 14 facts, 0% for 70B. The interference is localized to fact-encoding subspaces and doesn't damage general output distribution.

Validated on Llama-3.1-8B (BF16) and Llama-3.1-70B (NF4) on 2×H100 80GB.

**Papers** (all open access on Zenodo):
1. [Sleep-Wake Consolidation](https://doi.org/10.5281/zenodo.18778760) — LoRA sleep-wake prototype on 3B
2. [The Alignment Tax](https://doi.org/10.5281/zenodo.18778762) — RLHF inverse scaling discovery
3. [Dual-System Memory](https://doi.org/10.5281/zenodo.18778764) — MEMIT+LoRA, null-space constraints
4. [Sleeping LLM: Two-Phase](https://doi.org/10.5281/zenodo.18778766) — SWS+REM, per-fact staged consolidation
5. [Sleep-Wake Convergence](https://doi.org/10.5281/zenodo.18778768) — MEMIT-only, convergence proof

**Code**: [github.com/vbario/sleeping-llm](https://github.com/vbario/sleeping-llm) — full system + 33 experiment scripts

Happy to discuss the null-space constraint mechanism, the alignment tax, or the biological CLS framing.

---

## Reddit r/LocalLLaMA

### Title
I built sleep for local LLMs — model learns facts from conversation during wake, maintains them during sleep. Runs on MacBook Air.

### Body
After 4 months of research (5 papers, 122 development notes), I have a working system where a local LLM forms persistent memories from conversation — no RAG, no database. The facts are in the weights. After restart with an empty context window, the model knows things it learned from talking to you.

**How it works:**

- **Wake**: You chat normally. The system extracts facts and injects them into MLP weights via MEMIT (Mass-Editing Memory in Transformers). Single forward pass, instant recall. No training.
- **Sleep**: Type `/sleep` and the system audits every stored fact, refreshes degraded ones with null-space constraints (so fixing one memory doesn't break others), and prunes excess.

**What runs where:**

| Hardware | Model | Facts | Notes |
|---|---|---|---|
| MacBook Air M3, 8GB | Llama-3.2-3B-4bit | ~15 | Works today, sleep ~5 min |
| 2×H100 80GB | Llama-3.1-8B | 30 | 100% recall after sleep |
| 2×H100 80GB | Llama-3.1-70B | 60 | 100% recall, 0% PPL impact |

**The most surprising finding**: LoRA-based memory consolidation (my original approach) completely fails at 70B. RLHF alignment creates a behavioral prior that overrides LoRA-injected knowledge — 0% recall despite successful training. The effect gets *worse* with model size. I had to abandon LoRA entirely. MEMIT with sleep maintenance turned out to be simpler and more robust.

**The biological parallel**: This is basically CLS theory (Complementary Learning Systems) from neuroscience. Wake = hippocampal fast encoding. Sleep = consolidation. The system even has a "drowsiness signal" — it monitors how many facts are degraded and knows when it needs sleep.

**Setup:**
```
git clone https://github.com/vbario/sleeping-llm.git && cd sleeping-llm
pip3 install -r requirements.txt
python3 -m src.main
```

First run downloads the model (~1.8 GB). Requires Apple Silicon Mac with macOS 14+.

**Papers** (all free on Zenodo): [Paper 1](https://doi.org/10.5281/zenodo.18778760) | [Paper 2](https://doi.org/10.5281/zenodo.18778762) | [Paper 3](https://doi.org/10.5281/zenodo.18778764) | [Paper 4](https://doi.org/10.5281/zenodo.18778766) | [Paper 5](https://doi.org/10.5281/zenodo.18778768)

Happy to answer questions. The `notes/` directory has 122 numbered research notes if you want to see the full journey including every failure.
