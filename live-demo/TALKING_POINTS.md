# Quick-Reference Talking Points

## Soundbites (pick your favorites)

- "Every AI today has amnesia. We fixed that."
- "This is a brain that learns while it sleeps — running on a $999 laptop."
- "Not RAG. Not retrieval. The model literally rewired its own weights."
- "LoRA training plus weight fusion — the model literally rewires itself while it sleeps."
- "Graduation means the model knows it without being told. Like you know your own name."

## For Skeptics

**"Isn't this just fine-tuning?"**
> "Traditional fine-tuning is a sledgehammer — hours of training on massive datasets. We use LoRA with augmented Q&A pairs and priority weighting, then fuse the adapter into base weights. It takes about 60 seconds on a laptop. And the graduation system tells us when a fact is actually absorbed — no guessing."

**"Why not just use RAG?"**
> "RAG is ctrl+F on a database. It doesn't change what the model knows — it just pastes context. Our model actually KNOWS the fact. Remove the prompt, ask a different question, it still answers. That's the difference between looking something up and remembering it."

**"Can this scale?"**
> "On a 3B model we cap at about 8-10 facts before interference. Larger models (8B, 70B) have far more capacity. The approach scales with model size — more parameters means more room for facts."

**"What about catastrophic forgetting?"**
> "Priority-weighted training and the graduation system handle this. Facts are trained with augmented paraphrases so the learning is distributed across the model, not concentrated. And the graduation test catches regressions — if a previously learned fact stops being recalled, it gets un-graduated and goes back into the system prompt."

## Technical Gems (for engineer audience)

1. **The echo trap**: When question=answer=value="Jazzy Mike is a saxophone player", the graduation test just checks if the model echoes back the statement. It always passes — but the model can't answer real questions. Fixed by extracting just the key value ("saxophone player") and testing with generated questions.

2. **Chat template matters**: Training data must use the same chat-template format the model sees during inference. Early versions trained on raw text — the model learned facts in a format it never actually uses during conversation.

3. **Question paraphrasing is key**: Training on a single Q&A pair per fact leads to brittle recall. Generating 4 paraphrases per fact dramatically improves generalization — the model answers varied phrasings, not just the exact training question.

4. **Priority weighting**: Not all facts are equal. High-surprise facts get more training weight, which helps the model allocate its limited capacity to the most important information.

## Numbers That Impress

| What | Number |
|------|--------|
| Model parameters | 3 billion |
| Quantization | 4-bit (1.5GB on disk) |
| Hardware | MacBook Air M3, 8GB RAM |
| Sleep cycle time | ~90 seconds |
| LoRA training | 40 iterations, ~60 seconds |
| LoRA target layers | 8 of 26 |
| LoRA rank | 16 |
| Cost | $0 (runs locally, no API) |
