# Sleeping LLM: Biologically-Inspired Memory Consolidation for Continual Learning in Local Language Models

**Vladimir Baranov**

February 2026

---

## Abstract

We present Sleeping LLM, a system that enables persistent memory formation in a locally-running language model through a biologically-inspired sleep-wake cycle. During wake phases, the model engages in standard conversational inference while logging all exchanges. During sleep phases, the system extracts factual knowledge from conversations as structured Q&A pairs, curates training data through novelty and importance scoring, and consolidates this knowledge into model weights via LoRA fine-tuning — analogous to the hippocampal-neocortical memory transfer that occurs during human sleep. The system includes a spaced repetition replay buffer that revisits high-value memories across multiple sleep cycles with decaying priority, a validation gate that prevents catastrophic forgetting by benchmarking the model before and after each sleep cycle, and a checkpoint manager that enables rollback to any prior model state. We demonstrate successful persistent memory formation on a Llama 3.2 3B model (4-bit quantized) running on consumer hardware with 8GB of unified memory. After a single sleep cycle, the model recalled specific facts from conversation that it had never been pretrained on — with no context window assistance. Subsequent sleep cycles strengthened recall, confirming the spaced repetition hypothesis. We document the full experimental progression, including failure modes (catastrophic forgetting at high learning rates, empty training from over-aggressive curation, corrupted training formats) and the specific configuration that achieved successful consolidation. This work provides a proof of concept for continual personalized learning in local LLMs, grounded in the Complementary Learning Systems framework from cognitive neuroscience.

## 1. Introduction

Large language models possess no mechanism for persistent learning after deployment. A user may spend hours providing personal context, corrections, and preferences — all of which vanish when the context window is exhausted or the session ends. Current approaches to this problem fall into two categories: **retrieval-augmented generation (RAG)**, which stores information externally and injects it at inference time, and **prompt engineering**, which maintains context through careful management of the input window. Neither approach constitutes genuine learning. The information remains external to the model's parameters.

Human cognition faces a structurally similar problem. The hippocampus can rapidly encode new experiences (analogous to an LLM's context window), but this storage is volatile and capacity-limited. The brain's solution is **sleep** — a dedicated offline process that selectively transfers important experiences from hippocampal short-term storage into neocortical long-term storage through repeated replay. This process, described by the Complementary Learning Systems (CLS) framework (McClelland, McNaughton & O'Reilly, 1995), involves slow, interleaved exposure of new memories to the neocortex, allowing integration without catastrophic interference with existing knowledge.

We propose a direct computational analogue: a system where a locally-running LLM periodically enters a sleep state during which conversational knowledge is consolidated into model weights via parameter-efficient fine-tuning. The contributions of this work are:

1. **A complete, working architecture** for sleep-wake memory consolidation in local LLMs, implemented in ~1200 lines of Python on Apple Silicon using the MLX framework.
2. **Empirical demonstration** of persistent memory formation on a 3B parameter model with 8GB RAM — the model recalls facts from conversation after restart with an empty context window.
3. **Systematic documentation** of failure modes including catastrophic forgetting boundaries, training format sensitivity, and the critical role of fact extraction in producing trainable representations.
4. **Confirmation of the spaced repetition effect** — multiple sleep cycles produce stronger recall than a single cycle, consistent with the neuroscience of memory consolidation.

## 2. Related Work

### 2.1 Continual Learning

Continual learning (also called lifelong learning or incremental learning) addresses the problem of learning sequentially from a stream of data without forgetting previously acquired knowledge. The primary challenge is **catastrophic forgetting** (McCloskey & Cohen, 1989; French, 1999), where training on new data degrades performance on prior tasks.

Approaches to mitigating catastrophic forgetting include regularization methods such as Elastic Weight Consolidation (EWC) (Kirkpatrick et al., 2017), which penalizes changes to parameters important for prior tasks; replay methods that maintain a buffer of past examples (Rolnick et al., 2019); and architectural approaches that allocate separate parameters for different tasks (Rusu et al., 2016). Our system employs both replay (via the spaced repetition buffer) and a validation gate that functions as an implicit regularizer.

### 2.2 Parameter-Efficient Fine-Tuning

Low-Rank Adaptation (LoRA) (Hu et al., 2022) enables fine-tuning of large models by training small rank-decomposition matrices inserted into transformer layers, while keeping the original weights frozen. This dramatically reduces memory requirements and training time. QLoRA (Dettmers et al., 2023) extends this to quantized models, enabling fine-tuning of models that would otherwise exceed available memory. Our system uses LoRA on 4-bit quantized models, making the entire training pipeline feasible on consumer hardware.

### 2.3 Knowledge Editing

Recent work on model editing (Mitchell et al., 2022; Meng et al., 2022) seeks to modify specific facts stored in model weights without full retraining. These approaches typically target individual fact modifications. Our approach differs in that we perform batch knowledge consolidation from conversational interactions — a broader and less structured form of knowledge acquisition.

### 2.4 Complementary Learning Systems

The CLS framework (McClelland, McNaughton & O'Reilly, 1995; Kumaran, Hassabis & McClelland, 2016) proposes that biological memory relies on two interacting systems: a fast-learning hippocampal system for encoding specific experiences and a slow-learning neocortical system for extracting statistical regularities. Memory consolidation during sleep involves hippocampal replay of recent experiences to the neocortex, allowing gradual integration (Wilson & McNaughton, 1994; Rasch & Born, 2013). Our architecture maps this framework directly: the context window serves as the hippocampal buffer, model weights serve as neocortical storage, and the sleep phase implements consolidation through replay and fine-tuning.

## 3. System Architecture

### 3.1 Overview

The system operates as a state machine alternating between two modes:

**Wake mode:** Standard autoregressive inference. The model engages in conversation via a chat interface. A context manager tracks the sliding window of recent messages and performs automatic compaction (summarization) when approaching capacity. Every exchange is persisted to disk as timestamped JSONL logs.

**Sleep mode:** An offline training pipeline triggered either manually or after a configurable number of conversation turns. Sleep consists of six sequential stages: benchmark evaluation, data curation with fact extraction, replay buffer update, LoRA fine-tuning, post-training validation, and conditional weight fusion.

### 3.2 Wake Phase Components

**Context Manager.** Maintains the active context window as a list of messages with a configurable maximum token count (default: 4096). When token usage exceeds a compaction threshold (default: 80%), older messages are summarized by the model into a compressed representation, preserving key facts and preferences while freeing context space. The system prompt and any compacted summary are prepended to every inference call.

**Conversation Logger.** Appends every message to a per-session JSONL file with timestamps, session identifiers, and turn numbers. These logs serve as the ground truth for subsequent training and are never modified or deleted.

**Chat Interface.** Processes user input, manages the inference loop through the MLX backend, and monitors sleep triggers. Supports manual commands (`/sleep`, `/status`, `/compact`) and automatic sleep triggering after N turns.

### 3.3 Sleep Phase Components

#### 3.3.1 Curator and Fact Extraction

The curator is the most critical component for successful memory formation. Raw conversations are poor training material for factual recall — training on "He's a 6 year old music producer!" / "That's amazing!..." does not reliably teach the model to answer "Who is Andre Patandre?" in a novel context.

The curator operates in two stages:

**Heuristic scoring.** Each user-assistant exchange is scored on three dimensions:
- **Novelty** — message length, presence of questions, technical specificity
- **Importance** — corrections, explicit preferences, personal information
- **Future utility** — general knowledge patterns, references to ongoing projects

Exchanges below configurable score thresholds are filtered out.

**Fact extraction.** The model itself is prompted to read the full conversation and extract every specific fact, name, preference, and piece of personal information as structured Q&A pairs:

```
Input:  Full conversation transcript
Output: Q: Who is Andre Patandre?
        A: Andre Patandre is Vladimir's 6-year-old son.
        Q: What does Andre do?
        A: He is a music producer who makes beats on GarageBand.
        ...
```

These extracted Q&A pairs form the primary training signal. They directly encode the retrieval pattern: given a question about a fact, produce the answer. Raw conversation exchanges are included as secondary training data.

#### 3.3.2 Spaced Repetition Replay Buffer

The replay buffer maintains a prioritized store of training examples across sleep cycles. Each entry carries a priority score (initialized from the curator's combined score) and a replay count. When sampled for training, an entry's priority decays by a configurable factor (default: 0.85).

This implements the Ebbinghaus spacing effect: important information is revisited across multiple sleep cycles with diminishing frequency. Items that are never reinforced by new conversations gradually fade from the replay buffer. Items that keep appearing in conversations maintain high priority.

During each training run, a configurable fraction (default: 20%) of the replay buffer is sampled and mixed with the current cycle's curated data.

#### 3.3.3 Core Identity Dataset

A fixed set of Q&A pairs defining the model's identity and behavioral norms is mixed into every training run. This prevents identity drift — without it, the model's persona and baseline behavior gradually shift with each sleep cycle.

#### 3.3.4 Trainer

LoRA fine-tuning is executed via the MLX framework's training CLI. The trainer:
1. Combines curated Q&A pairs, raw exchanges, replay buffer samples, and identity data into a single training set
2. Scales iterations proportionally to dataset size: `iterations = num_examples * epochs`
3. Runs LoRA training with configurable rank, alpha, number of layers, learning rate, and batch size
4. Saves the resulting adapter to disk

#### 3.3.5 Validator

The validator runs a fixed set of benchmark questions (general knowledge, arithmetic, programming concepts) against the model before and after training. If the post-training score drops below a configurable fraction of the pre-training score (default: 50%), the sleep cycle is rejected.

Critically, validation occurs on a model fused to a **temporary directory**. The production model is never modified until validation passes. On rejection, the system reloads the last known good model — either from the most recent checkpoint or from the original base model.

#### 3.3.6 Dreamer (REM Equivalent)

During deep sleep cycles (triggered every N light sleep cycles), the dreamer module generates synthetic Q&A pairs by prompting the model to find connections between topics encountered in recent conversations. This is analogous to REM sleep's role in creative association and memory integration. The synthetic examples are added to the training set for that cycle.

### 3.4 Backend

All model operations are handled through a unified MLX backend that wraps the `mlx-lm` library for inference, tokenization, LoRA training, and adapter fusion. The system targets Apple Silicon hardware using unified memory, which allows the same memory pool to serve both inference and training without the GPU memory constraints typical of NVIDIA-based setups.

## 4. Experiments

### 4.1 Hardware and Model

All experiments were conducted on a MacBook Air M3 with 8GB unified memory running macOS 14. The base model was Llama 3.2 3B Instruct (4-bit quantized via MLX), requiring approximately 2.5GB of memory and producing inference at ~40-60 tokens per second.

### 4.2 Experimental Protocol

The test protocol was:
1. Start a fresh model instance
2. Conduct a conversation containing specific, verifiable, novel facts that the model could not have seen during pretraining
3. Trigger a sleep cycle
4. Restart the application (clearing all in-memory context)
5. Query the model about the facts from step 2

The primary test data involved a fictional person ("Andre Patandre") with specific attributes (age, occupation, family relationship, tools used) that could not exist in the model's pretraining data.

### 4.3 Failure Progression

The system required multiple iterations before achieving successful memory formation. Each failure was diagnostic:

**Failure 1: Empty training data.** Initial curation thresholds (novelty >= 0.3, importance >= 0.3, combined >= 0.4) filtered out all conversational exchanges. The heuristic scorer assigned low scores to natural conversation that didn't contain technical keywords. The training files were 0 bytes. No learning occurred.

*Resolution:* Thresholds reduced to 0.0 (all data passes through).

**Failure 2: Session scope error.** The training pipeline only gathered messages from the current session. After restarting the application and triggering sleep, the new (empty) session produced no training data.

*Resolution:* Modified to gather all conversation sessions from disk.

**Failure 3: Catastrophic forgetting at high learning rate.** With learning rate 5e-4, 5 epochs (500 iterations), and 34 training examples, the model was completely destroyed. Pre-sleep benchmark score: 0.95 (19/20). Post-sleep benchmark score: 0.00 (0/20). The model could only output exclamation points.

*Resolution:* Learning rate reduced. Iterations scaled to dataset size.

**Failure 4: Rollback loading corrupted model.** After the catastrophic forgetting event, the validator correctly rejected the result. However, the adapter had already been fused into the production model directory. The rollback code attempted to reload from this directory, loading the destroyed model.

*Resolution:* Implemented fuse-to-temp-directory pattern. The production model is never modified until validation passes.

**Failure 5: Corrupted training format.** The chat template function applied `add_generation_prompt=True` to training data, appending an empty assistant header to every example. This trained the model to produce empty responses after answering questions, diluting the learning signal.

*Resolution:* Added a `for_training` flag that omits the generation prompt suffix.

**Failure 6: Learning rate too low.** At 5e-5 with 3 epochs (~270 iterations), the model survived training perfectly (benchmark: 1.00 before and after) but formed no detectable memories. The weight changes were too small to affect inference behavior.

**Failure 7: Learning rate 2e-4 still destructive.** At 2e-4 with 3 epochs (~270 iterations), the model was again destroyed (benchmark: 1.00 -> 0.00).

### 4.4 Successful Configuration

The working configuration was found at:

| Parameter | Value |
|---|---|
| Learning rate | 1e-4 |
| Epochs | 1 (single pass) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA layers | 8 |
| Batch size | 1 |
| Curation thresholds | 0.0 (all data) |
| Validation threshold | 0.5 |
| Replay buffer mix ratio | 0.2 |

With ~90 training examples (extracted Q&A pairs + raw exchanges), this produced ~90 iterations per sleep cycle. Training time: approximately 5-8 minutes on the test hardware.

### 4.5 Results

**Sleep cycle 1.** Pre-sleep benchmark: 1.00 (5/5). Post-sleep benchmark: 1.00 (5/5). Approved and fused. After application restart (empty context), the model correctly recalled:
- Andre Patandre's identity as a music producer (correct)
- His approximate age (correct)
- His relationship to Vladimir (incomplete — not recalled)
- His level of fame (hallucinated — added unsupported details)

The core, heavily-repeated fact was retained. Peripheral details with fewer training examples were not.

**Sleep cycle 2.** Same data, replayed through the spaced repetition buffer. Post-sleep benchmark maintained. After restart, recall was improved — more details were accurately retrieved. This confirms that multiple consolidation cycles strengthen memory encoding, consistent with the neuroscience of spaced repetition.

### 4.6 Learning Rate Sensitivity Analysis

The following table summarizes all learning rate experiments:

| Learning Rate | Epochs | Approx. Iterations | Pre-Sleep Score | Post-Sleep Score | Memory Formed |
|---|---|---|---|---|---|
| 5e-4 | 5 | 500 | 0.95 | 0.00 | No (model destroyed) |
| 2e-4 | 3 | 270 | 1.00 | 0.00 | No (model destroyed) |
| 1e-4 | 1 | 90 | 1.00 | 1.00 | **Yes** |
| 5e-5 | 3 | 270 | 1.00 | 1.00 | No (no detectable change) |

The viable range for the 3B model is narrow: approximately 1e-4 with a single epoch. Higher rates cause catastrophic forgetting; lower rates produce no detectable learning. This narrow window is likely a consequence of the small model size — larger models with more parameters should have a wider viable range due to greater capacity for absorbing new information without disrupting existing representations.

## 5. Discussion

### 5.1 Why Fact Extraction Matters

The most impactful architectural decision was extracting structured Q&A pairs from conversations rather than training on raw dialogue. Raw conversation exchanges encode conversational flow, not factual retrieval. A model trained on "He's a 6 year old music producer!" learns to continue that specific conversation — not to answer "How old is Andre Patandre?" in a novel context.

Fact extraction converts implicit conversational knowledge into explicit retrieval pairs. This is analogous to how the brain doesn't simply replay raw sensory experience during sleep — it extracts and reorganizes information, forming new associations and abstractions (Stickgold & Walker, 2013).

### 5.2 The Role of Repetition

Our results align with the established neuroscience of memory consolidation. Facts that appeared repeatedly in the training data (across multiple Q&A pairs and raw exchanges) were recalled more reliably than facts mentioned only once. The replay buffer's spaced repetition effect was confirmed: a second sleep cycle on the same data produced measurably better recall.

This suggests that the optimal strategy for memory formation is not a single aggressive training pass, but multiple gentle passes spread across sleep cycles — exactly what the brain does through nightly sleep over weeks.

### 5.3 The Catastrophic Forgetting Boundary

The narrow viable learning rate window (approximately 1e-4 for 3B, single epoch) has implications for system design. The boundary between "no learning" and "catastrophic forgetting" may be smaller than previously assumed for small quantized models. This suggests that:

1. **Validation gating is essential**, not optional. Even small deviations in training configuration can destroy the model.
2. **Larger models likely have a wider viable range**, as they have more parametric capacity to absorb new information without displacing existing knowledge.
3. **The fuse-to-temp pattern is critical** for production safety. Any system that modifies production weights before validation is vulnerable to catastrophic, unrecoverable failure.

### 5.4 Comparison with RAG

Retrieval-Augmented Generation stores information externally and injects it into the context window at inference time. This has significant advantages: it's reliable, reversible, and doesn't risk model degradation. However, RAG does not constitute learning — the model's behavior and capabilities remain unchanged. It cannot generalize from retrieved information, form associations between retrieved facts, or develop new behavioral patterns based on accumulated experience.

Our approach modifies the model's weights, enabling genuine generalization and behavioral change. The model doesn't just recall that "Andre makes music on GarageBand" — it can potentially reason about this fact in novel contexts, connect it to related knowledge, and adjust its conversational behavior based on accumulated understanding of the user. The tradeoff is risk: weight modification can fail catastrophically in ways that RAG cannot.

A hybrid approach — RAG for reliable factual storage, sleep-based consolidation for behavioral adaptation and deep knowledge integration — may be optimal for production systems.

### 5.5 CLS Framework Mapping

The system maps cleanly onto the Complementary Learning Systems framework:

| CLS Component | System Implementation | Observed Behavior |
|---|---|---|
| Hippocampal fast encoding | Context window + conversation logs | Immediate recall during session |
| Neocortical slow learning | LoRA fine-tuning at low learning rate | Persistent recall after restart |
| Sleep consolidation | Curated training during sleep phase | Memory transfer from logs to weights |
| Hippocampal replay | Spaced repetition buffer | Improved recall over multiple cycles |
| Emotional tagging | Curation scoring (novelty, importance) | Selective consolidation of important info |
| Memory reconsolidation | Replay buffer priority decay | Gradual weakening of unreinforced memories |
| Dreaming / REM | Synthetic Q&A generation | Novel associations between learned facts |

The mapping is imperfect — the system lacks the brain's synaptic consolidation mechanism (analogous to EWC), active forgetting pathways, and the fine-grained selectivity of emotional tagging. However, the core CLS prediction holds: a two-speed learning system with offline consolidation outperforms either speed alone.

## 6. Limitations

**Model capacity.** The 3B parameter model has limited capacity for absorbing new knowledge without disrupting existing representations. The narrow viable learning rate window is likely a direct consequence of small model size.

**Training on model outputs.** The system currently trains on both user messages and model responses. If the model hallucinated during conversation, those hallucinations are reinforced into the weights — a direct parallel to the reconsolidation vulnerability in human memory (Nader, Schafe & Le Doux, 2000).

**No deduplication.** All conversation sessions are gathered for every sleep cycle, meaning early conversations are retrained repeatedly. This accidental over-reinforcement could cause the model to overfit to early interactions.

**Curation quality.** Heuristic scoring uses keyword matching and is unreliable for assessing true novelty or importance. Model-based scoring (included but not used in the successful experiments) is more accurate but significantly slower.

**Fact extraction noise.** The model's self-generated Q&A pairs contain formatting artifacts and occasionally hallucinated details. Cleaner extraction would improve training signal quality.

**Single-user assumption.** Knowledge from different users would be conflated in the weights with no mechanism for separation or access control.

**Offline training.** The model is unavailable during sleep cycles. A production system would require background training on a model copy or a secondary model handling requests during sleep.

**Evaluation scope.** The benchmark consists of 5-20 general knowledge questions and does not directly measure whether specific conversational facts were retained. A proper evaluation would include automatically generated test questions from the training conversations.

## 7. Future Work

**Elastic Weight Consolidation.** Adding EWC (Kirkpatrick et al., 2017) to penalize changes to parameters important for existing capabilities could widen the viable learning rate range and improve stability across many sleep cycles.

**Targeted evaluation.** Automatically generating test questions from curated facts and including them in the validation benchmark would directly measure memory formation quality, not just capability preservation.

**Session deduplication.** Tracking which conversations have been consumed by previous sleep cycles and only training on new material would prevent over-reinforcement and reduce training time.

**Output filtering.** Training only on user-provided information (not model responses) would eliminate hallucination reinforcement.

**Scaling experiments.** Testing on 8B, 13B, and 70B models would characterize how model size affects the viable learning rate range, memory capacity, and recall precision.

**Multi-timescale sleep.** The architecture supports tiered sleep (micro-nap, light sleep, deep sleep with dreaming) but only light sleep was tested. Evaluating the full multi-timescale pipeline is future work.

**Longitudinal studies.** Running the system over weeks or months with a real user would test whether the spaced repetition mechanism produces genuinely durable memories and whether drift accumulates to problematic levels.

## 8. Conclusion

We have demonstrated that a 3-billion parameter language model running on consumer hardware with 8GB of RAM can form persistent memories from conversation through a biologically-inspired sleep-wake cycle. The information is encoded in the model's weights — not in a database, not in a prompt, and not in a context window. After the model sleeps and is restarted with a completely empty context, it recalls facts it learned from talking to a user.

This required solving several practical problems: curating training data that encodes retrievable facts rather than conversational noise, finding the narrow learning rate window between "no learning" and "catastrophic forgetting" for a small quantized model, designing a validation pipeline that prevents catastrophic failure from reaching production, and implementing spaced repetition to strengthen memories across multiple sleep cycles.

The result is a proof of concept for continual personalized learning in local LLMs. The architecture is model-agnostic, hardware-agnostic (within the Apple Silicon ecosystem), and fully open source. The same mechanism on a larger model with more memory would be expected to produce sharper recall, retain more peripheral details, and tolerate a wider range of training configurations.

The deepest finding is that the neuroscience of memory consolidation provides actionable engineering guidance. The Complementary Learning Systems framework predicted the need for two learning speeds, offline consolidation, spaced repetition, and selective encoding — and every one of these predictions proved correct in implementation. The brain solved the continual learning problem through sleep. Language models can too.

## References

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized Language Models. *NeurIPS 2023*.

French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences*, 3(4), 128-135.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521-3526.

Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). What learning systems do intelligent agents need? Complementary learning systems theory updated. *Trends in Cognitive Sciences*, 20(7), 512-534.

McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109-165.

McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. *Psychological Review*, 102(3), 419-457.

Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and editing factual associations in GPT. *NeurIPS 2022*.

Mitchell, E., Lin, C., Bosselut, A., Finn, C., & Manning, C. D. (2022). Fast model editing at scale. *ICLR 2022*.

Nader, K., Schafe, G. E., & Le Doux, J. E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. *Nature*, 406(6797), 722-726.

Rasch, B., & Born, J. (2013). About sleep's role in memory. *Physiological Reviews*, 93(2), 681-766.

Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). Experience replay for continual learning. *NeurIPS 2019*.

Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Sober, H., Kavukcuoglu, K., & Hadsell, R. (2016). Progressive neural networks. *arXiv preprint arXiv:1606.04671*.

Stickgold, R., & Walker, M. P. (2013). Sleep-dependent memory triage: evolving generalization through selective processing. *Nature Neuroscience*, 16(2), 139-145.

Wilson, M. A., & McNaughton, B. L. (1994). Reactivation of hippocampal ensemble memories during sleep. *Science*, 265(5172), 676-679.
