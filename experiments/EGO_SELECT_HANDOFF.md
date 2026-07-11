# EGO-SELECT Handoff — Continue the Matrix on This Machine

**Written 2026-07-11 on the M3 Air; run migrating to the M2 Mac Mini (8GB).**
You are picking up a pre-registered, partially-executed experiment. Read
`notes/131-ego-selector-experiment` (the master spec, incl. amendments) and
`notes/132-ego-publishable-outcomes` (what we hope to publish) before acting.

## Current state (all committed on branch `ego-select`)

- Pre-registration §7.4 steps 1–6 COMPLETE: plan → frozen artifacts → corpus →
  implementation → freeze (green: 43/43 acquirable, leak demonstrated + fix
  verified, oracle denominators pre=8.0 post=6.0) → pilot (parse-rate 0%
  fallback all four judge conditions; 4/4 quick runs green).
- Full matrix (§7.4 step 7) PARTIAL: **7 of 24 cells complete**, results in
  `experiments/results/ego_<cond>_s<seed>.json` (committed). Completed: all 8
  seed-41 conditions except `oracle` — i.e. random, surprise, borrowed, judge,
  judge_rules, ego_static, ego_full @ s41. The `*_quick.json` files are pilot
  artifacts, not matrix cells.
- The matrix was killed 3× on the M3 Air by memory pressure (jetsam kills the
  python process at model load when the machine is in use) — hence migration.

## To resume the matrix (the only step you need)

```bash
cd <repo> && git checkout ego-select
caffeinate -is ./experiments/run_ego_matrix.sh 2>&1 | tee -a matrix_mini.log
```

- The runner is resumable and failure-tolerant: cells with a valid result JSON
  are skipped; it clears `__pycache__` between runs; ~13–16 min/cell on an M3
  Air, expect similar or slightly slower on the M2 (~4–5h for the remaining 17
  cells). Keep the machine otherwise idle (8GB — model load loses jetsam fights).
- If a cell is interrupted mid-run: `rm -rf data/ego_exp/<cond>_<seed>` before
  relaunching, then relaunch the same command.
- First run on this machine will download `mlx-community/Llama-3.2-3B-Instruct-4bit`
  (~1.8GB). Requirements: `pip install -r requirements.txt` (mlx, mlx_lm 0.29.x).

## After the matrix completes (§7.4 step 8)

1. `python3 experiments/analyze_ego.py` — computes M1–M10, gates G1–G4, runs
   the pre-registered statistics (run-level permutation within seed, seed
   sign-consistency 3/3, Holm correction), and emits a verdict-table skeleton.
2. Check gates FIRST: if G3 (oracle sensitivity) fails, the experiment is
   INCONCLUSIVE per pre-registration — do not interpret contrasts.
3. Write `notes/133-ego-select-results` in house style (title + `====`
   underline, date, prose sections) mirroring §3's predictions table with
   SUPPORTED / FALSIFIED / SKIP / INCONCLUSIVE per P0/V1/P2/P2b/P2a/P3–P7,
   trajectory tables, per-cell survival heatmap, ρ comparison, comparison
   against notes/111 and notes/62 baselines, and a Files section. Report every
   null. notes/132 pre-registered what each outcome means for publication.
4. Commit results + note on `ego-select`. Commit message convention: prefix
   `EGO-SELECT:`, end with `Co-Authored-By:` line (see `git log` on this branch).
5. Findings already banked regardless of matrix outcome (report them in 133):
   the HallucinationFirewall self-grounding leak + curator-gate fix (freeze
   log), the speaker-centric/stochastic extractor characterization (notes/131
   §7.1.3 amendments), the chat-template requirement for valuation calls, and
   the incumbent selector's novelty degeneracy (§2 scope 6).

## Context you'd otherwise lack

- All new behavior is config-gated and OFF by default; the matrix configs in
  `experiments/configs/3b_ego_*.yaml` turn on: admission_gate, uniform_reps,
  prune_on_graduation=false, train_timeout=1800, surprise threshold 1.1
  (forced consolidation timing), memit+micro_sleep off.
- Tests: `python3 -m pytest tests/ -q` → 34 pass; 27 failures are PRE-EXISTING
  debt (old `ledger=` kwarg / removed `extract_template` API), not from this
  branch.
- The pre-registration lives in git history: rules/prompts committed BEFORE
  the corpus; amendments are explicit dated commits. Do not rewrite history.
