# Sleeping LLM — Live Demo Script

## Concept (30 seconds)

The Sleeping LLM gives a small local model (3B parameters, runs on a MacBook Air) **persistent memory through weight modification** — not RAG, not context stuffing, but actual changes to the neural network's weights during "sleep" cycles. Facts are extracted from conversation, buffered, then consolidated into the model via LoRA training. After enough sleep cycles, the model *knows* the facts without being told — they've been absorbed into its weights.

---

## Setup

- **Model**: Llama-3.2-3B-Instruct-4bit (MLX, Apple Silicon)
- **Web UI**: http://localhost:8000
- **Clean slate**: Factory reset before demo so ledger is empty

---

## Act 1 — Conversation & Fact Extraction (~3 minutes)

### What to do

Have a natural conversation, dropping 5 personal facts across several messages:

> "Hey, do you know who Jimmy Jimz is?"
>
> "Jimmy Jimz is a music producer from Seattle. He makes rap beats."
>
> "His real name is James, he's 28 years old."
>
> "He just released an album called Nightcrawler."

### What to show / point out

1. **Right sidebar — "Buffered" section**: After each message, facts appear as orange dots with short statements. Point out that these are being extracted *by the model itself* — the model reads the conversation and lists what new concrete facts it learned.

2. **Extraction quality**: Facts should appear as clean declarative sentences:
   - "Jimmy Jimz is a music producer from Seattle"
   - "Jimmy Jimz makes rap beats"
   - "Jimmy Jimz's real name is James"
   - "Jimmy Jimz is 28 years old"
   - "Jimmy Jimz released an album called Nightcrawler"

3. **Deduplication**: If the model re-extracts a fact it already knows (e.g. "Seattle" mentioned again), it gets silently filtered out. The buffer won't show duplicates.

4. **Click the token count** in the left sidebar under "Context" to open the context viewer. Show the audience that the facts are being injected into the system prompt:
   ```
   Things you remember about the user:
   - Jimmy Jimz is a music producer from Seattle
   - Jimmy Jimz makes rap beats
   ...
   ```
   This is Tier 0 recall — instant, but uses context window space.

5. **Consolidate**: Click "Consolidate Now" in the right sidebar (or let the surprise estimator trigger it automatically). Facts move from "Buffered" (orange) to "Consolidated" (green, stage 0). They're now persisted in the fact ledger.

---

## Act 2 — First Sleep Cycle (~2 minutes)

### What to do

Click the **Sleep** button in the left sidebar. A modal overlay appears showing the 5-step sleep pipeline in real time.

### What the audience sees — step by step

#### Step 1: Health Check
- Measures baseline **perplexity** (PPL) on a reference text
- This is the "before" snapshot — if sleep makes the model worse, we'll know

#### Step 2: Curating
- Scans unconsumed conversation sessions
- Re-extracts facts from the full conversation history
- Deduplicates against existing ledger (typically adds 0-2 new facts)
- Marks sessions as consumed so they won't be re-processed

#### Step 3: LoRA Consolidation
- **This is the core innovation.** All non-graduated facts become training data
- Each fact is formatted as a Q&A pair for the chat template
- Priority-weighted: important facts get more training iterations
- LoRA adapter is trained (8 layers, ~80 iterations)
- Adapter is **fused into the base model weights** — this is a permanent change
- The fused model is saved and reloaded

#### Step 4: Graduation Test
- For each fact: remove it from the system prompt, then ask the model to recall it
- If the model can recall the fact *without being told* → advance stage (0→1)
- If it fails → retreat to stage 0
- At stage 3 (after 3 successful sleep cycles), the fact **graduates**:
  - Removed from the system prompt entirely
  - Pruned from all active lists
  - The LoRA-modified weights carry the knowledge now

**Key talking point**: "After this first sleep, facts advance from stage 0 to stage 1. The model is starting to learn them, but it still needs the system prompt as a safety net."

#### Step 5: Validation
- Measures PPL again on the same reference text
- Compares before/after: if PPL increased too much (>15%), something went wrong
- Typically PPL stays flat or improves slightly

### After sleep

Click "Continue" to dismiss the overlay. Show:
- Consolidated facts now say "stage 1" in their metadata
- The model still knows the facts (ask it: "What do you know about Jimmy Jimz?")

---

## Act 3 — Second & Third Sleep Cycles (~3 minutes)

### What to do

Run two more sleep cycles back-to-back (click Sleep, wait ~2 min, dismiss, repeat). Optionally add 1-2 more conversational turns between them to keep it natural.

### What to point out

- **Second sleep**: Facts advance stage 1 → 2. Model is getting more confident.
- **Third sleep**: Facts advance stage 2 → 3 = **GRADUATED**.
  - The graduation step will show: "X graduated and absorbed, Y remain"
  - Graduated facts disappear from the consolidated list in the sidebar
  - They've been pruned — the model's weights now carry this knowledge permanently

### The payoff moment

After the third sleep, **open the context viewer** (click Tokens). Show the audience:
- The graduated facts are **no longer in the system prompt**
- The context window is lighter — those facts freed up space

Now ask the model: "Tell me everything you know about Jimmy Jimz."

**The model recalls the facts from its weights alone** — no system prompt injection, no RAG, no retrieval. The knowledge is in the neural network.

---

## Act 4 — Proof It's Real (~1 minute)

### Factory reset test (optional, high-impact)

To really prove this isn't tricks:

1. Note the current model path in the sidebar (should show the fused model)
2. Ask a recall question — model answers correctly
3. Show the context viewer — no facts in prompt
4. Explain: "The facts are in the weights now. Even if I clear all the metadata, the model still knows."

---

## Key Talking Points

### Why not just use RAG?
RAG retrieves text and stuffs it into the context window. The model doesn't *know* anything — it's reading a cheat sheet. Sleeping LLM actually modifies the model's weights so the knowledge becomes intrinsic. This is closer to how biological memory consolidation works during sleep.

### Why sleep?
- Real-time weight editing (MEMIT) was tried and removed — it corrupts the model on small architectures
- LoRA training takes time (~2 min on an 8GB MacBook Air)
- The sleep metaphor is accurate: knowledge transfers from short-term (system prompt) to long-term (weights) during an offline consolidation phase

### The 3B model limit
Our experiments showed 3B models can reliably hold ~8-10 facts via LoRA before recall degrades. That's why there's a hard cap of 8 facts. Larger models (70B) can hold significantly more — but the point is this runs locally on a laptop.

### The graduation mechanism
Facts go through 3 sleep cycles (stages 0→1→2→3) before graduating. Each cycle tests: "Can the model recall this fact without being told?" This prevents premature graduation — the model must consistently demonstrate knowledge before we trust it.

### What makes this different from fine-tuning?
Fine-tuning is a one-shot process on a static dataset. Sleeping LLM is a **continuous, incremental** system — it learns from every conversation, buffers knowledge, consolidates during sleep, and graduates facts when they're stable. It's lifelong learning, not batch training.

---

## Timing

| Segment | Duration |
|---------|----------|
| Concept intro | 30s |
| Conversation + extraction | 3 min |
| First sleep cycle | 2.5 min |
| Second sleep | 2 min |
| Third sleep + graduation | 2.5 min |
| Proof / recall test | 1 min |
| **Total** | **~12 min** |

---

## Troubleshooting

- **Model produces junk facts**: The 3B model occasionally hallucinates. If a bad fact appears, just continue — the dedup and junk filters catch most issues, and the demo flow still works.
- **Sleep takes too long**: LoRA training on 8GB M3 is ~90-130s. This is normal. Use the wait to explain what's happening at each step.
- **Fact doesn't graduate**: If a fact fails the recall test, it retreats to stage 0. This is the system working correctly — mention that the model needs more training on that fact.
- **PPL increases**: Mild increase (<15%) is acceptable and won't block the merge. Mention that the validation gate protects against catastrophic forgetting.
