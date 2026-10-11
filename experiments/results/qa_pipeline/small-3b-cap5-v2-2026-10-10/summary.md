# QA pipeline eval — small-3b-cap5-v2-2026-10-10

- date: 2026-10-10 04:34:25
- backend: auto
- model: mlx-community/Llama-3.2-3B-Instruct-4bit
- hardware: Apple M5 Pro, 48 GB RAM, macOS-26.5.2-arm64-arm-64bit
- size: small {"n_facts": 20, "per_session": 5, "seeds": 2, "n_updates": 2, "usage_sample": 5, "bleed_neighbors": 2}
- seeds: 2
- elapsed: 346.4s

## sleep condition (mean±std over seeds)

| cycle | graduated | grad. recall | retention | bleed | usage | updates | ppl drift | rollbacks |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.00±0.00 |   —   |   —   | 0.05±0.05 |   —   |   —   | 0.04±0.01 | 0  |
| 1 | 0.00±0.00 |   —   |   —   | 0.25±0.25 |   —   |   —   | 0.06±0.01 | 0  |
| 2 | 0.00±0.00 |   —   |   —   | 0.00±0.00 |   —   | 0.00±0.00 | 0.06±0.01 | 1 ['bleed'] |
| 3 | 0.00±0.00 |   —   |   —   | 0.00±0.00 |   —   | 0.00±0.00 | 0.08±0.03 | 0  |

## prompt_only baseline (all facts in the prompt, no training)

| cycle | prompt recall | usage | updates | bleed |
|---|---|---|---|---|
| 0 | 1.00±0.00 | 1.00±0.00 |   —   | 0.00±0.00 |
| 1 | 1.00±0.00 | 1.00±0.00 |   —   | 0.00±0.00 |
| 2 | 0.95±0.05 | 1.00±0.00 | 0.75±0.25 | 0.00±0.00 |
| 3 | 0.95±0.03 | 0.90±0.10 | 1.00±0.00 | 0.00±0.00 |

