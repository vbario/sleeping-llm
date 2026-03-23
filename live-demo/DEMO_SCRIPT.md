# Live Demo Script: Sleeping LLM — A Brain That Learns While It Sleeps

## The Hook (30 seconds)

"What if your AI could remember things you told it yesterday — not because it saved a note, but because it literally rewired its own brain? Today I'm going to show you a 3-billion parameter model running on an 8GB MacBook Air that learns personal facts during conversation and consolidates them into its weights while it sleeps — just like a biological brain."

---

## Act 1: The Problem (2 minutes)

**Show:** Start the web UI. Ask the model "Who is Jazzy Mike?"
**Expected:** Model has no idea. Generic response.

**Talking points for audience:**
- Every LLM you use today has amnesia. ChatGPT, Claude — they don't remember you between sessions.
- RAG (retrieval) is a band-aid: it's ctrl+F on a database, not real memory.
- Fine-tuning is expensive and slow — you can't retrain GPT-4 every time a user tells you their name.
- **The question:** Can a small local model actually LEARN new facts and carry them in its weights?

**For engineers:**
- This is the "catastrophic forgetting" problem — neural nets forget old knowledge when learning new things.
- We're running Llama 3.2 3B (4-bit quantized) entirely on-device. No cloud. No API calls.
- The system uses Apple's MLX framework for on-device inference and training.

---

## Act 2: Teaching Facts (3 minutes)

**Show:** Tell the model facts about a fictional person:
- "Jazzy Mike is a saxophone player from Toronto, Canada."
- "He has a hit song called Swing the Bag."
- "He's 34 years old and allergic to shellfish."

**Show the Facts panel** updating in real-time as facts are extracted.

**Talking points for audience:**
- The model extracts facts from natural conversation automatically — you don't fill out a form.
- Right now, the model remembers because the facts are injected into its "working memory" (system prompt).
- This is like your short-term memory — it works, but it's fragile. Clear the context and it's gone.

**For engineers:**
- Fact extraction uses the model itself as the extractor (no separate NER pipeline).
- Facts are stored as Q&A pairs in a FactLedger (JSON on disk).
- The "value" field contains just the key term (e.g., "saxophone player" not the full statement) — this matters for graduation testing later.
- System prompt injection = Tier 0 memory. Instant recall, but limited by context window (4096 tokens).

---

## Act 3: The Sleep Cycle (5 minutes) — THE MAIN EVENT

**Show:** Trigger `/sleep` and narrate each step as it progresses.

### Step 1: Health Check
"First, it measures its own perplexity — how confused is it? This is the baseline."

### Step 2: Curation
"It reads back through the conversation and extracts facts it hasn't seen before."

### Step 3: LoRA Consolidation
"Now the actual brain surgery begins."

**LoRA Training:**
"It creates training data from the facts — but not just echo training. It generates multiple question phrasings so the model learns to answer varied questions, not just parrot statements. Then it trains a low-rank adapter and fuses it directly into the base weights."

### Step 4: Graduation Test
"For each fact, it removes that fact from the system prompt and asks the model a question. If the model can still answer from its weights alone — the fact graduates. It's been absorbed."

### Step 5: Validation
"Final sanity check — did we break the model? Perplexity comparison."

**For engineers:**
- LoRA = low-rank adaptation. Trains a small adapter (rank 16) on 8 target layers.
- Training data uses chat-template-formatted Q&A pairs so the adapter fires during actual chat inference.
- Multiple question paraphrases per fact improve generalization — the model doesn't just memorize one phrasing.
- After training, the adapter is fused permanently into the base weights (no separate adapter file needed).

---

## Act 4: The Proof (2 minutes)

**Show:** After sleep completes:
1. Clear the conversation context (or restart the server)
2. Ask "Who is Jazzy Mike?" — model should now know from weights alone
3. Ask "Where is he from?" — tests different phrasing
4. Ask "What's his hit song?" — tests a different fact

**Talking points for audience:**
- "No notes. No database lookup. The model literally knows this now — it's in the weights."
- "This is running on an 8GB MacBook Air. No cloud. No GPU cluster."

**For engineers:**
- The graduation test uses a DIFFERENT question than what was trained on — this tests generalization, not memorization.
- LoRA distributed adaptation across 8 target layers provides robust recall.
- Inspired by the brain's sleep consolidation: replaying memories to integrate them into long-term storage.

---

## Act 5: The Architecture (1 minute — with diagram)

**Show:** The architecture diagram (see `architecture.svg`).

"The system has four memory tiers — just like your brain has working memory, short-term memory, and long-term memory."

---

## Closing: Why This Matters (1 minute)

- "Every AI assistant today is a stranger. It meets you fresh every time."
- "Imagine an AI doctor that remembers your medical history in its weights. An AI tutor that knows which concepts you struggle with. An AI companion that actually knows you."
- "This is running on a $999 laptop with a 3B model. The same approach scales to larger models with more capacity."
- "We're open-source. The code is on GitHub. Go build on it."

---

## Contingency: If Facts Don't Graduate

If the graduation test fails, explain:
- "The 3B model has limited capacity — about 8-10 facts via LoRA before interference."
- "This is expected. Larger models (8B, 70B) hold significantly more."
- "The system prompt still provides instant recall — graduation is the aspirational goal."
- Show the system prompt with injected facts as proof the system works at Tier 0.

---

## Key Numbers to Mention

| Metric | Value |
|--------|-------|
| Model | Llama 3.2 3B Instruct (4-bit) |
| Hardware | MacBook Air M3, 8GB RAM |
| Model size on disk | ~1.5 GB |
| LoRA training time | ~60 seconds per sleep |
| Max facts (LoRA, 3B) | ~8-10 before interference |
| Framework | Apple MLX |
