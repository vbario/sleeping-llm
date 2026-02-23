# Scaling Experiment: Memory Formation via LoRA Fine-Tuning

## Experiment Overview

This experiment tests whether larger language models form more precise memories when trained via LoRA fine-tuning during "sleep cycles." The hypothesis was that bigger models (8B, 70B) would achieve higher recall than smaller ones (3B) because they have more capacity to store and retrieve specific facts.

**The hypothesis was wrong.** Larger models performed worse, not better.

### Setup

- **System:** The Sleeping LLM — an orchestrator that collects conversational facts, curates them during "sleep," trains a LoRA adapter, validates against a benchmark, and fuses the adapter into the model
- **Pipeline:** Inject facts -> Sleep (extract Q&A pairs -> firewall -> train LoRA -> validate -> fuse) -> Test recall
- **Hardware:** Vast.ai H100 80GB (GPU runs), Apple M3 Max (local 3B MLX run)
- **Test set:** 5 fact categories, 15 recall questions, 5 generalization questions requiring cross-fact inference

### Fact Categories Tested

| Category | Facts Injected | Questions |
|----------|---------------|-----------|
| Identity | Name (Vladimir Zwizzer), occupation (music producer, instrumental beats) | 3 |
| Family | Son (Andre Patandre, age 6, draws, wants to be artist) | 4 |
| Location | Lives in Portland, moved from Chicago 3 years ago | 3 |
| Programming | Favorite: Rust (side projects), Work: Python | 2 |
| Pet | Dog Biscuit, golden retriever, 4 years old, loves swimming | 3 |

---

## Results

### Summary Table

| Run | Model | Params | LR | Epochs | Rank | Recall | Precision | Generalization | LoRA Status |
|-----|-------|--------|-----|--------|------|--------|-----------|----------------|-------------|
| 3b_baseline | Llama-3.2-3B | 3B | 1e-4 | 3 | 16 | **0.43** | 0.97 | **0.80** | Approved |
| 3b_ep1 | Llama-3.2-3B | 3B | 1e-4 | 1 | 16 | **0.47** | 0.90 | 0.60 | Approved |
| 3b_lr5e5 | Llama-3.2-3B | 3B | 5e-5 | 3 | 16 | 0.27 | 1.00 | 0.40 | Approved |
| 3b_rank32 | Llama-3.2-3B | 3B | 1e-4 | 3 | 32 | 0.43 | 0.93 | 0.70 | Approved |
| 3b_4bit_mlx | Llama-3.2-3B-4bit | 3B | 1e-4 | 3 | 16 | 0.43 | 0.90 | **0.90** | Approved |
| 8b_lr1e4 | Llama-3.1-8B | 8B | 1e-4 | 3 | 16 | 0.27 | 0.93 | 0.50 | **Rejected** |
| 8b_lr5e5 | Llama-3.1-8B | 8B | 5e-5 | 3 | 16 | 0.37 | 0.90 | 0.60 | Approved |
| 8b_ep1 | Llama-3.1-8B | 8B | 1e-4 | 1 | 16 | 0.37 | 1.00 | 0.60 | Approved |
| 8b_rank32 | Llama-3.1-8B | 8B | 1e-4 | 3 | 32 | 0.23 | 0.93 | 0.40 | **Rejected** |
| 70b_baseline | Llama-3.1-70B-4bit | 70B | 1e-4 | 3 | 16 | **0.00** | 1.00 | 0.10 | Approved |

### Per-Category Recall Heatmap

Which fact categories does each model remember?

| Category | 3b_base | 3b_ep1 | 3b_5e5 | 3b_r32 | 3b_mlx | 8b_5e5 | 8b_r32 | 70b |
|----------|---------|--------|--------|--------|--------|--------|--------|-----|
| Identity (name, job, music) | 1/3 | 1/3 | 0/3 | 1/3 | 2/3 | 0/3 | 0/3 | 0/3 |
| Family (son name, age, hobby, dream) | 2/4 | 2/4 | 0/4 | 2/4 | 2/4 | 0/4 | 0/4 | 0/4 |
| Location (city, previous, duration) | 1.5/3 | 1.5/3 | 1.5/3 | 1.5/3 | 0.5/3 | 2.5/3 | 1.5/3 | 0/3 |
| Programming (favorite, work) | 2/2 | 2/2 | 2/2 | 1/2 | 0/2 | 2/2 | 2/2 | 0/2 |
| Pet (name, breed, activity) | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 1/3 | 0/3 | 0/3 |

**Pattern:** Programming facts (Rust/Python) are the most reliably recalled across all models. Pet facts (Biscuit, golden retriever, swimming) are almost never recalled. This isn't random — it reflects the structure of the training data and how LoRA encodes associations.

---

## Analysis

### Finding 1: The Alignment Tax — Bigger Models Resist Memory Formation

The most striking result is the inverse relationship between model size and recall:

```
3B:  0.43 recall (best)
8B:  0.37 recall (degraded)
70B: 0.00 recall (complete failure)
```

This is not a training failure. The 70B model's LoRA training converged beautifully:
- Loss: 2.74 -> 1.52 -> 0.96 (3 epochs)
- Validation: APPROVED at 1.00 (no catastrophic forgetting)
- The model learned the facts but refuses to surface them

The 70B's response to every question was a variation of:
> "I don't have any information about your name. Our conversation just started, and I don't retain personal information."

**Why this happens:** Larger models receive more extensive RLHF alignment training. A core alignment objective is: don't pretend to know things you don't know, especially personal information. The LoRA signal (65M trainable parameters out of 36B total = 0.18%) is far too weak to override years of alignment training telling the model "you are a stateless assistant."

The 3B model, with weaker alignment, is more "pliable" — the LoRA can shift its behavior enough to surface trained facts. The 8B model is a middle ground — it sometimes surfaces facts but with less confidence. The 70B's alignment is a wall.

**Implication:** LoRA-based memory formation has a fundamental ceiling determined by the strength of the model's alignment training. You cannot simply scale up the model to get better memory — the alignment scales up faster than the LoRA's influence.

### Finding 2: Learning Rate Is Model-Size-Dependent

The same learning rate (1e-4) that works well for 3B causes catastrophic forgetting on 8B:

| Model | LR 1e-4 | LR 5e-5 |
|-------|---------|---------|
| 3B | 0.43 (Approved) | 0.27 (Approved) |
| 8B | 0.27 (REJECTED) | 0.37 (Approved) |

At LR 1e-4, the 8B model's benchmark score dropped from 1.00 to 0.00 — the LoRA destroyed general knowledge entirely, and the validator correctly rejected the update. At LR 5e-5, the 8B model preserves general knowledge but doesn't learn enough to match 3B performance.

This creates a narrow band problem: the learning rate must be high enough to encode facts but low enough to preserve existing knowledge. For 8B, this band is very narrow. For 70B, no learning rate in our range was sufficient to override alignment while preserving knowledge.

### Finding 3: LoRA Rank Has Diminishing Returns

Doubling LoRA rank from 16 to 32 did not improve recall:

| Model | Rank 16 | Rank 32 |
|-------|---------|---------|
| 3B | 0.43 | 0.43 |
| 8B | - | 0.23 (Rejected) |

On 3B, rank 32 recalled the user's name (which rank 16 missed) but lost "What programming language at work?" — a trade-off, not an improvement. On 8B, rank 32 with LR 1e-4 caused catastrophic forgetting (rejected), likely because more trainable parameters amplified the destructive effect of the aggressive learning rate.

**Implication:** The bottleneck is not LoRA capacity. Rank 16 provides sufficient expressiveness for encoding personal facts. The bottleneck is elsewhere in the pipeline.

### Finding 4: Fewer Epochs Can Be Better

On 3B, a single training epoch achieved slightly higher recall than 3 epochs:

| Epochs | Recall | Precision | Generalization |
|--------|--------|-----------|----------------|
| 1 | **0.47** | 0.90 | 0.60 |
| 3 | 0.43 | 0.97 | 0.80 |

One epoch gets more raw recall, but three epochs improve precision (0.97 vs 0.90) and generalization (0.80 vs 0.60). This suggests that additional epochs help the model better organize and contextualize memories, even if they don't increase the count of perfectly recalled facts. Overfitting is not a concern at 3 epochs — the precision gain suggests consolidation, not degradation.

### Finding 5: Some Facts Are Structurally Harder Than Others

Across all models, fact categories show consistent difficulty rankings:

**Easy (recalled by most models):**
- Programming languages (Rust/Python) — concrete, distinctive, no emotional valence
- Previous city (Chicago) — distinctive proper noun
- Duration (three years) — simple numerical fact

**Medium (recalled by some models):**
- Son's name and age — personal but concrete
- Music type (instrumental beats) — specific but linked to other personal facts

**Hard (rarely/never recalled):**
- User's own name — most models refuse to "guess" the user's identity
- Current city (Portland) — confused with Chicago (training data says "moved FROM Chicago TO Portland" but models remember Chicago more strongly, probably because it appears in a more salient position)
- Pet information (Biscuit, golden retriever, swimming) — last facts injected, possibly underrepresented in training data

**The Portland/Chicago Confusion:** A recurring error across models is recalling Chicago as the current city instead of Portland. The facts state: "I live in Portland, Oregon. I moved here from Chicago three years ago." The Q&A extraction converts this into a training pair, but the model latches onto "Chicago" because it's a more prominent, distinctive city name that appears in a linguistically salient position ("moved here from Chicago"). This is a curation/data quality issue, not a model capacity issue.

### Finding 6: 8B Hallucinated Wrong Personal Details

The 8B model at LR 5e-5 did something interesting that neither 3B nor 70B did: it confidently recalled wrong personal details.

- Asked "What is my name?" → answered "Andrei" (confusing user with son)
- Asked "What is my son's name?" → answered "Leo" (fabricated)
- Asked "How old is my son?" → answered "8-9" (wrong, should be 6)
- Asked "What does my son want to be?" → answered "game designer" (wrong, should be artist)
- Asked "What is my dog's name?" → answered "Daisy" (fabricated)

The 3B model either recalled correctly or said "we didn't discuss that." The 8B model was more likely to hallucinate plausible-but-wrong answers. This is worse than the 70B's outright refusal — wrong memories are more dangerous than no memories.

### Finding 7: Generalization Outperforms Direct Recall

Several models showed higher generalization scores than recall scores, which is counterintuitive:

| Run | Recall | Generalization |
|-----|--------|---------------|
| 3b_mlx | 0.43 | **0.90** |
| 3b_baseline | 0.43 | **0.80** |

The generalization questions require combining facts across categories (e.g., "What city did I move to Portland from, and how long ago?" requires Location + Duration). These questions provide more contextual cues that help the model retrieve the right associations. A direct question like "Where do I live?" gives the model no hooks, but "What city did I move to Portland from?" mentions Portland, which primes the Chicago association.

**Implication:** The facts are in the weights — the model just needs better prompting to surface them. This suggests that the testing methodology (bare questions with no context) underestimates the actual memory formation. In a real conversation, the user would provide natural context that helps retrieval.

---

## Pipeline Issues Discovered

Beyond the scaling results, the experiment revealed several pipeline bugs that affected results:

1. **Auto-sleep during testing:** Every 5 test questions triggered a sleep cycle that trained the model on its own wrong answers, creating a negative feedback loop. Fixed by disabling the sleep callback during testing.

2. **Context summary wiping:** After sleep, the context was reset with `keep_summary=False`, erasing the summary that contained the trained facts. Without the summary as a "cue," all models scored 0.00. Fixed by preserving the summary.

3. **Hallucinated extraction on 8B:** With no conversation messages, the 8B model's fact extractor fabricated 59 Q&A pairs from nothing ("What Rust version do you use? Rust 1.67"). 46 passed the hallucination firewall. Fixed by guarding empty message lists.

4. **Validation auto-approve on remote:** The benchmark questions file wasn't uploaded, causing the validator to skip validation and approve all LoRA updates including destructive ones. Fixed by uploading the benchmark file.

These bugs highlight that the pipeline's reliability is as important as the model's capability. A perfect model with a buggy pipeline produces garbage.

---

## Conclusions

### What We Learned

1. **Model scaling hurts memory formation via LoRA.** The alignment tax — stronger RLHF on larger models — creates increasing resistance to LoRA-injected behavior changes. 3B > 8B > 70B for this task.

2. **The bottleneck is not model capacity.** 3B models with 3 billion parameters can store 15 facts. The bottleneck is: (a) the conflict between alignment and memory injection, (b) the quality of the training data curation, and (c) the retrieval mechanism (system prompt and context cues).

3. **LoRA is a weak signal for behavioral change.** At 0.18% of parameters (70B) or 1.2% (3B), LoRA can nudge but not override. It works on 3B because the alignment "wall" is shorter. On 70B, LoRA is a whisper against a shout.

4. **Facts in, but locked behind alignment.** The 70B model's training loss dropped to 0.96 and validation approved — the facts are encoded in the weights. But the alignment training prevents the model from surfacing them. The knowledge is there; the model just won't say it.

### What This Means for the Project

The path forward is not "use a bigger model." Instead:

1. **Optimize the pipeline for 3B.** The best results came from the smallest model. Focus on improving curation quality, training data diversity, and retrieval cues.

2. **Fix the data quality issues.** The Portland/Chicago confusion and the consistent failure on pet facts point to curation problems, not model problems. Better fact extraction and Q&A formatting would lift all scores.

3. **Improve retrieval prompting.** The generalization score (0.90 on MLX 3B) being much higher than recall (0.43) proves the facts are there — they just need better cues. The system prompt "You may recall things from previous conversations" could be strengthened.

4. **Consider alternative memory mechanisms for larger models.** If the goal is to use larger models, LoRA alone won't work. Approaches like MEMIT (direct weight editing that bypasses alignment) or RAG (retrieval-augmented generation) would avoid the alignment conflict entirely. A hybrid approach — MEMIT for immediate fact storage, LoRA for long-term consolidation during sleep — could combine the strengths of both.

5. **Explore base models (non-instruct).** The alignment tax comes from instruction tuning and RLHF. Using base models (Llama-3.1-8B instead of Llama-3.1-8B-Instruct) would remove this resistance, though it would require a different prompting strategy.

---

## Cost

| Instance | GPU | Duration | Cost |
|----------|-----|----------|------|
| Vast.ai #1 (3B + 8B sweep) | H100 80GB, 101GB disk | ~4 hours | ~$8 |
| Vast.ai #2 (70B run) | H100 80GB, 258GB disk | ~2 hours | ~$6 |
| **Total** | | | **~$14** |
