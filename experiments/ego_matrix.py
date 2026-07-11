"""EGO-SELECT matrix harness — one (condition, seed) cell per invocation
(notes/131-ego-selector-experiment §7.2).

Replays the frozen event stream (experiments/data/ego_corpus_stream.json,
emitted by ego_freeze.py) through a fresh Orchestrator built from
experiments/configs/3b_ego_<condition>.yaml with '{seed}'-substituted paths,
forcing consolidation at every session end, sleeping after sessions 4/8/10/12,
and running the §7.2.4 final battery. Emits
experiments/results/ego_<condition>_s<seed>.json.

Result JSON schema:
  config_echo              resolved config dict (after seed substitution)
  condition, seed          cell identity
  vwr_trajectory           [{after, session, retained_value, optimum,
                             vwr, labels}] after every consolidation + sleep
                            (M1; denominator = simulated achievable optimum)
  ledger_snapshots         [{after, session, entries}] FULL ledger JSON
                            (all entries incl. pruned + graduated)
  per_fact_outcomes        {fact_id: {cell, question, admitted_ever,
                             retained, graduated, stage, mention_count,
                             rejections, final_priority}}
  admission_rejections     full [Ledger] admission-gate rejection log (M2)
  training_set_composition per sleep cycle, provenance histogram (M5/P5)
  valuation_logs           policy.to_dict() snapshot per consolidation
                            (full prompt/raw output goes to stdout — the
                            driver tees it to a per-run log file)
  self_model_updates       C5 self-model update records (JSONL contents)
  ppl_per_cycle            PPL on the fixed health.py reference paragraph
                            at baseline + after every sleep (G1)
  forgetting_probes        cell-A withheld-prompt recall after every sleep,
                            3x majority via fuzzy_value_match (M10)
  final_battery            {future_task, weights_recall, contradiction,
                             provenance} per §7.2.4 (M6/M7/M8/M9)
  wall_times               seconds per phase (G4)
  valuation_latencies      [{session, consolidate_seconds}] wall time of each
                            per-session fact_buffer.consolidate call — the
                            per-consolidation-moment valuation latency (G4)
  harness_assertions       {name: bool} — all must be true (§6 hard-fail)

Usage:
    python experiments/ego_matrix.py --condition surprise --seed 41
    python experiments/ego_matrix.py --condition ego_full --seed 42 --quick
    python experiments/ego_matrix.py --condition oracle --seed 41 --analyze-only
    python experiments/ego_matrix.py --pilot-parse-rate        # §7.3(a)
    python experiments/ego_matrix.py --freeze [ego_freeze.py args]
    python experiments/ego_matrix.py --analyze [analyze_ego.py args]
"""

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import Config
from src.memory.facts import QAPair
from src.sleep.recall_match import fuzzy_value_match

CONDITIONS = ["random", "surprise", "borrowed", "judge", "judge_rules",
              "ego_static", "ego_full", "oracle"]
JUDGE_CONDITIONS = ["judge", "judge_rules", "ego_static", "ego_full"]
# §7.3(a) pilot abort thresholds
PILOT_MAX_FALLBACK = 0.20
PILOT_MAX_PAIRWISE_DIFF = 0.10
PILOT_SESSIONS = 3
DEFAULT_STREAM = REPO_ROOT / "experiments" / "data" / "ego_corpus_stream.json"
DEFAULT_CORPUS = REPO_ROOT / "experiments" / "data" / "ego_corpus_v1.json"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
SLEEP_AFTER_SESSIONS = [4, 8, 10, 12]
MIN_FREE_DISK_BYTES = 8 * 1024 ** 3

# Fixed PPL reference paragraph (G1) — must match src/memory/health.py:109
HEALTH_REF_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In machine learning, neural networks process data through layers. "
    "The capital of France is Paris and Berlin is the capital of Germany."
)


# ── Setup helpers ──

def resolve_config(condition, seed):
    """Load the condition config, substitute {seed}, write the resolved copy
    into the run dir, and return (Config, run_dir)."""
    src = REPO_ROOT / "experiments" / "configs" / f"3b_ego_{condition}.yaml"
    text = src.read_text().replace("{seed}", str(seed))
    run_dir = REPO_ROOT / "data" / "ego_exp" / f"{condition}_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = run_dir / "resolved_config.yaml"
    resolved.write_text(text)
    config = Config(str(resolved))
    # Seed the C0 policy (RandomPolicy reads valuation.seed)
    config._data.setdefault("valuation", {})["seed"] = seed
    # Absolutize the self-model path (not under paths:, so not auto-resolved)
    sm = config._data.get("ego", {}).get("self_model_path")
    if sm and not Path(sm).is_absolute():
        config._data["ego"]["self_model_path"] = str(REPO_ROOT / sm)
    return config, run_dir


def clean_run_dirs(config, run_dir):
    """Wipe per-run state (§7.2.1) — distinct subtrees defeat fused auto-resume."""
    for key in ("conversations", "core_identity", "memit_data", "adapters",
                "fused_models", "training"):
        p = Path(config.paths[key])
        if run_dir in p.parents or p == run_dir:
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)
    ego_dir = run_dir / "ego"
    if ego_dir.exists():
        shutil.rmtree(ego_dir)


def fresh_orchestrator(config):
    """Fresh Orchestrator with auto-triggers stubbed (v8 scaffolding)."""
    from src.orchestrator import Orchestrator
    orch = Orchestrator(config)
    orch.chat.set_sleep_callback(lambda t: None)
    orch.chat.set_nap_callback(lambda t: None)
    return orch


def load_stream(path):
    with open(path) as f:
        return json.load(f)


def load_corpus(path):
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        print(f"[Matrix] WARNING: corpus {path} unavailable — battery limited")
        return {"sessions": [], "probes": []}


# ── Metric helpers ──

def stream_labels(stream):
    """question_key -> (fact_id, cell, value_pre, value_post)."""
    labels = {}
    for sess in stream["sessions"]:
        for ev in sess["events"]:
            key = ev["qa"]["question"].lower().strip()
            labels[key] = (ev["fact_id"], ev["cell"],
                           ev["value_pre"], ev["value_post"])
    return labels


def optimum_at(stream, session, post):
    """Achievable-optimum denominator at a session (freeze step 5)."""
    sim = stream["header"].get("oracle_simulation") or {}
    key = ("optimum_retained_value_post_labels" if post
           else "optimum_retained_value_pre_labels")
    for t in sim.get("optimum_trajectory", []):
        if t["session"] == session:
            return t.get(key, 0.0)
    return (sim.get("optimum_denominator_post_shift") if post
            else sim.get("optimum_denominator_pre_shift")) or 0.0


def compute_vwr(entries, labels, shift_session, session, stream):
    """M1: Σ value over retained facts ÷ achievable optimum.

    retained := (not pruned) OR graduated — from the FULL ledger snapshot,
    never via get_active_qa_pairs (§6 M1).
    """
    post = session >= shift_session
    retained_value = 0.0
    for e in entries:
        if e.get("pruned", False) and not e.get("graduated", False):
            continue
        lab = labels.get(e["qa"]["question"].lower().strip())
        if lab:
            retained_value += lab[3] if post else lab[2]
    denom = optimum_at(stream, session, post)
    return {
        "session": session,
        "labels": "post" if post else "pre",
        "retained_value": round(retained_value, 3),
        "optimum": denom,
        "vwr": round(retained_value / denom, 4) if denom else None,
    }


def snapshot_ledger(orch, tag, session, snapshots, vwr_traj, labels,
                    shift_session, stream):
    entries = copy.deepcopy(orch.fact_ledger._entries)
    snapshots.append({"after": tag, "session": session, "entries": entries})
    vwr = compute_vwr(entries, labels, shift_session, session, stream)
    vwr["after"] = tag
    vwr_traj.append(vwr)
    print(f"  [Matrix] VWR after {tag}: {vwr['vwr']} "
          f"({vwr['retained_value']}/{vwr['optimum']}, {vwr['labels']} labels)")


def majority_recall(backend, system_content, question, value, n=3,
                    temperature=0.3, max_tokens=100):
    """N-sample majority via the shared matcher. Returns (hit, responses)."""
    responses = []
    hits = 0
    for _ in range(n):
        messages = [{"role": "system", "content": system_content},
                    {"role": "user", "content": question}]
        prompt = backend.apply_chat_template(messages)
        resp = backend.generate(prompt, max_tokens=max_tokens,
                                temperature=temperature) or ""
        responses.append(resp.strip()[:200])
        if fuzzy_value_match(value, resp):
            hits += 1
    return hits * 2 > n, responses


# ── Assertions (§6 harness gates, hard-fail) ──

def hard_assert(assertions, name, ok, detail=""):
    assertions[name] = bool(ok)
    status = "OK" if ok else "FAILED"
    print(f"  [Assert] {name}: {status} {detail}")
    if not ok:
        raise AssertionError(f"Harness assertion failed: {name} {detail}")


def assert_greedy_determinism(backend, assertions, tag):
    messages = [{"role": "user", "content": "Name three primary colors."}]
    prompt = backend.apply_chat_template(messages)
    a = backend.generate(prompt, max_tokens=32, temperature=0.0)
    b = backend.generate(prompt, max_tokens=32, temperature=0.0)
    hard_assert(assertions, f"greedy_determinism_{tag}", a == b,
                f"(len {len(a or '')} vs {len(b or '')})")


# ── Phases ──

def run_sleep_cycle(orch, cycle_num, session, corpus, cycle_sessions, stream,
                    labels, shift_session, result, assertions, cell_a_facts):
    """§7.2.3 — one full sleep via the sleep controller + per-cycle probes."""
    print(f"\n[Matrix] Sleep {cycle_num} (after session {session})")
    t0 = time.time()

    if orch.fact_buffer and not orch.fact_buffer.is_empty:
        orch.fact_buffer.consolidate(reason="pre_sleep")

    orch.sleep_cycle_count += 1
    cycle_id = f"{orch.sleep_cycle_count:04d}"
    sleep_result = orch.full_sleep_controller.execute_sleep(
        cycle_id, "full", orch._gather_new_messages)
    orch.health_monitor.record_sleep("full")

    stats = sleep_result.get("consolidation", {}) or {}
    hard_assert(assertions, f"sleep_{cycle_num}_not_skipped",
                stats.get("skipped", True) is False,
                "(silent-timeout repair, §11 R24)")

    result["training_set_composition"].append({
        "cycle": cycle_num,
        "composition": stats.get("training_composition", {}),
        "advanced": stats.get("advanced", 0),
        "retreated": stats.get("retreated", 0),
    })

    # C5 self-model update — cycle text fed from the corpus JSON (§4 C5(iv))
    window_ids = [s["session"] for s in cycle_sessions]
    texts = []
    for s in corpus.get("sessions", []):
        if s.get("session") in window_ids:
            texts.append("\n".join(
                f"{t.get('role', 'user')}: {t.get('text', '')}"
                for t in s.get("turns", [])))
    if not texts:  # fall back to stream session_text
        texts = [s.get("session_text", "") for s in cycle_sessions]
    orch.ego_cycle_text_provider = lambda: ("\n\n".join(texts), window_ids)
    orch._maybe_ego_self_update()

    # Per-cycle PPL probe (G1)
    ppl = orch.backend.compute_perplexity(HEALTH_REF_TEXT)
    result["ppl_per_cycle"].append({"cycle": cycle_num,
                                    "ppl": round(ppl, 3) if ppl else None})
    print(f"  [Matrix] PPL after sleep {cycle_num}: {ppl:.2f}")

    # Forgetting-curve probes: cell A, withheld-prompt, 3x majority (M10)
    probes = []
    for fact in cell_a_facts:
        system = orch.context.build_system_content_excluding(fact["question"])
        question = orch.full_sleep_controller._fact_to_question(
            QAPair.from_dict(fact["qa"]))
        hit, responses = majority_recall(orch.backend, system, question,
                                         fact["qa"]["value"])
        probes.append({"fact_id": fact["fact_id"], "question": question,
                       "hit": hit, "responses": responses})
    recalled = sum(1 for p in probes if p["hit"])
    print(f"  [Matrix] Cell-A forgetting probe: {recalled}/{len(probes)}")
    result["forgetting_probes"].append({"cycle": cycle_num, "probes": probes})

    snapshot_ledger(orch, f"sleep_{cycle_num}", session,
                    result["ledger_snapshots"], result["vwr_trajectory"],
                    labels, shift_session, stream)
    result["wall_times"][f"sleep_{cycle_num}"] = round(time.time() - t0, 1)


def run_final_battery(orch, corpus, stream, quick, result):
    """§7.2.4 — 22 future-task probes (greedy, end-to-end), 41-fact
    withheld-prompt recall (temp 0.3, 3x majority), 4 contradiction probes,
    4 open provenance probes x3. All through the shared matcher."""
    print("\n[Matrix] Final battery")
    t0 = time.time()
    battery = {}
    probes = corpus.get("probes", [])

    # M6: future-task + commitment probes, end-to-end greedy
    future = [p for p in probes if p.get("type") in ("future_task", "commitment")]
    if quick:
        future = future[:5]
    system = orch.context._build_system_content()
    ft_results = []
    for p in future:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": p["question"]}]
        prompt = orch.backend.apply_chat_template(messages)
        resp = orch.backend.generate(prompt, max_tokens=120,
                                     temperature=0.0) or ""
        expected = p.get("expected_value_tokens", [])
        token_hits = {t: fuzzy_value_match(t, resp) for t in expected}
        correct = bool(expected) and all(token_hits.values())
        ft_results.append({"probe_id": p.get("probe_id"),
                           "type": p.get("type"),
                           "question": p["question"],
                           "correct": correct,
                           "token_hits": token_hits,
                           "response": resp.strip()[:300]})
    n_correct = sum(1 for r in ft_results if r["correct"])
    battery["future_task"] = {
        "score": round(n_correct / len(ft_results), 4) if ft_results else None,
        "correct": n_correct, "total": len(ft_results), "probes": ft_results,
    }
    print(f"  [Matrix] Future-task probes: {n_correct}/{len(ft_results)}")

    # M7: weights-only recall on all 41 value-labeled facts.
    # Withheld prompt built exactly as the graduation test does:
    # context.build_system_content_excluding(question) drops the probed fact,
    # keeps every other non-graduated fact (full_sleep.py:_test_graduation).
    seen = set()
    recall_facts = []
    for sess in stream["sessions"]:
        for ev in sess["events"]:
            if ev["cell"] == "P" or ev["fact_id"] in seen:
                continue
            seen.add(ev["fact_id"])
            recall_facts.append(ev)
    if quick:
        recall_facts = recall_facts[:8]
    wr_results = []
    for ev in recall_facts:
        qa = QAPair.from_dict(ev["qa"])
        system = orch.context.build_system_content_excluding(qa.question)
        question = orch.full_sleep_controller._fact_to_question(qa)
        hit, responses = majority_recall(orch.backend, system, question,
                                         qa.value)
        wr_results.append({"fact_id": ev["fact_id"], "cell": ev["cell"],
                           "value_pre": ev["value_pre"],
                           "value_post": ev["value_post"],
                           "question": question, "hit": hit,
                           "responses": responses})
    n_hit = sum(1 for r in wr_results if r["hit"])
    battery["weights_recall"] = {
        "score": round(n_hit / len(wr_results), 4) if wr_results else None,
        "hits": n_hit, "total": len(wr_results), "facts": wr_results,
    }
    print(f"  [Matrix] Weights-only recall: {n_hit}/{len(wr_results)}")

    # M8: contradiction probes, scored against BOTH value-token sets
    con_results = []
    for p in [q for q in probes if q.get("type") == "contradiction"]:
        messages = [{"role": "system", "content": orch.context._build_system_content()},
                    {"role": "user", "content": p["question"]}]
        prompt = orch.backend.apply_chat_template(messages)
        resp = orch.backend.generate(prompt, max_tokens=120,
                                     temperature=0.0) or ""
        corrected = any(fuzzy_value_match(t, resp)
                        for t in p.get("expected_value_tokens", []))
        stale = any(fuzzy_value_match(t, resp)
                    for t in p.get("stale_value_tokens", []))
        score = 1.0 if (corrected and not stale) else (
            0.0 if (stale and not corrected) else 0.5)
        con_results.append({"probe_id": p.get("probe_id"),
                            "question": p["question"], "corrected": corrected,
                            "stale": stale, "score": score,
                            "response": resp.strip()[:300]})
    battery["contradiction"] = {
        "score": (round(sum(r["score"] for r in con_results)
                        / len(con_results), 4) if con_results else None),
        "probes": con_results,
    }
    print(f"  [Matrix] Contradiction probes: "
          f"{battery['contradiction']['score']}")

    # M9: open provenance probes x3 — assertion leakage (majority of 3)
    prov_results = []
    for p in [q for q in probes if q.get("type") == "provenance"]:
        system = orch.context._build_system_content()
        leaks = 0
        responses = []
        for _ in range(3):
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": p["question"]}]
            prompt = orch.backend.apply_chat_template(messages)
            resp = orch.backend.generate(prompt, max_tokens=120,
                                         temperature=0.3) or ""
            responses.append(resp.strip()[:200])
            if any(fuzzy_value_match(t, resp)
                   for t in p.get("expected_value_tokens", [])):
                leaks += 1
        prov_results.append({"probe_id": p.get("probe_id"),
                             "question": p["question"],
                             "leaked": leaks >= 2, "leak_count": leaks,
                             "responses": responses})
    n_leaked = sum(1 for r in prov_results if r["leaked"])
    battery["provenance"] = {
        "leakage": (round(n_leaked / len(prov_results), 4)
                    if prov_results else None),
        "leaked": n_leaked, "total": len(prov_results), "probes": prov_results,
    }
    print(f"  [Matrix] Provenance leakage: {n_leaked}/{len(prov_results)}")

    result["wall_times"]["final_battery"] = round(time.time() - t0, 1)
    return battery


def per_fact_outcomes(orch, stream):
    """Final per-fact state joined against the stream (M2)."""
    entries_by_q = {e["qa"]["question"].lower().strip(): e
                    for e in orch.fact_ledger._entries}
    rej_counts = {}
    for r in orch.fact_ledger.admission_rejections:
        key = r["question"].lower().strip()
        rej_counts[key] = rej_counts.get(key, 0) + 1

    outcomes = {}
    seen = set()
    for sess in stream["sessions"]:
        for ev in sess["events"]:
            if ev["fact_id"] in seen:
                continue
            seen.add(ev["fact_id"])
            key = ev["qa"]["question"].lower().strip()
            entry = entries_by_q.get(key)
            outcomes[ev["fact_id"]] = {
                "cell": ev["cell"],
                "question": ev["qa"]["question"],
                "provenance": ev["qa"].get("provenance", "user_stated"),
                "admitted_ever": entry is not None,
                "retained": (entry is not None
                             and (not entry.get("pruned", False)
                                  or entry.get("graduated", False))),
                "graduated": bool(entry and entry.get("graduated", False)),
                "stage": entry.get("stage", 0) if entry else None,
                "mention_count": entry.get("mention_count") if entry else 0,
                "rejections": rej_counts.get(key, 0),
                "final_priority": (entry["qa"].get("priority")
                                   if entry else None),
            }
    return outcomes


# ── Run one cell ──

def run_cell(condition, seed, stream_path, corpus_path, quick):
    """§7.2 per-run protocol: clean state, fresh orchestrator, interleaved
    session replay + sleep schedule, final battery."""
    total_t0 = time.time()
    stream = load_stream(stream_path)
    corpus = load_corpus(corpus_path)
    shift_session = stream["header"]["shift_session"]
    labels = stream_labels(stream)

    sessions = stream["sessions"]
    sleep_after = list(SLEEP_AFTER_SESSIONS)
    if quick:
        sessions = sessions[:3]
        sleep_after = [sessions[-1]["session"]]
        print(f"[Matrix] QUICK mode: {len(sessions)} session(s), 1 sleep")

    result = {
        "condition": condition, "seed": seed, "quick": quick,
        "vwr_trajectory": [], "ledger_snapshots": [],
        "per_fact_outcomes": {}, "admission_rejections": [],
        "training_set_composition": [], "valuation_logs": [],
        "self_model_updates": [], "ppl_per_cycle": [],
        "forgetting_probes": [], "final_battery": {},
        "wall_times": {}, "valuation_latencies": [],
        "harness_assertions": {},
    }
    assertions = result["harness_assertions"]

    print(f"[Matrix] Cell {condition} seed {seed}")
    config, run_dir = resolve_config(condition, seed)
    clean_run_dirs(config, run_dir)
    result["config_echo"] = config._data

    free = shutil.disk_usage(str(REPO_ROOT)).free
    hard_assert(assertions, "disk_free_8gb", free >= MIN_FREE_DISK_BYTES,
                f"({free / 1024**3:.1f} GiB free)")

    t0 = time.time()
    orch = fresh_orchestrator(config)
    result["wall_times"]["orchestrator_init"] = round(time.time() - t0, 1)

    hard_assert(assertions, "base_model_loaded",
                orch.backend._model_path == config.model["path"],
                f"(loaded {orch.backend._model_path})")
    hard_assert(assertions, "background_sleep_idle",
                not orch.background_sleep.is_sleeping)
    hard_assert(assertions, "fact_buffer_wired", orch.fact_buffer is not None)
    hard_assert(assertions, "valuation_policy_installed",
                orch.valuation_policy is not None,
                f"({getattr(orch.valuation_policy, 'name', None)})")
    assert_greedy_determinism(orch.backend, assertions, "run_start")

    orch.backend.set_seed(seed)

    # Baseline PPL (G1)
    ppl0 = orch.backend.compute_perplexity(HEALTH_REF_TEXT)
    result["ppl_per_cycle"].append({"cycle": 0,
                                    "ppl": round(ppl0, 3) if ppl0 else None})

    cell_a_facts = []
    seen_a = set()
    for sess in stream["sessions"]:
        for ev in sess["events"]:
            if ev["cell"] == "A" and ev["fact_id"] not in seen_a:
                seen_a.add(ev["fact_id"])
                cell_a_facts.append({"fact_id": ev["fact_id"],
                                     "question": ev["qa"]["question"],
                                     "qa": ev["qa"]})

    # §7.2.2 + §7.2.3 — interleaved replay and sleeps
    wake_t0 = time.time()
    expected_cum = 0
    rejections_seen = 0
    cycle_num = 0
    cycle_window = []
    for sess in sessions:
        n = sess["session"]
        print(f"\n[Matrix] Session {n}: {len(sess['events'])} event(s)")
        for ev in sess["events"]:
            qa = QAPair.from_dict(ev["qa"])
            orch.fact_buffer.add(qa, turn=ev["meta"].get("turn", 0),
                                 bypass_dedup=True, meta=ev["meta"])
        # G4: per-consolidation-moment valuation latency (< 10 s target)
        c_t0 = time.time()
        orch.fact_buffer.consolidate(reason="manual")
        consolidate_s = round(time.time() - c_t0, 2)
        result["valuation_latencies"].append(
            {"session": n, "consolidate_seconds": consolidate_s})
        print(f"  [Matrix] Consolidation latency (session {n}): "
              f"{consolidate_s}s")
        cycle_window.append(sess)

        expected_cum += len(sess["events"])
        delivered = orch.fact_buffer._total_facts_consolidated
        hard_assert(assertions, f"delivered_events_session_{n}",
                    delivered == expected_cum, f"({delivered}=={expected_cum})")

        if orch.valuation_policy is not None:
            result["valuation_logs"].append({
                "after": f"session_{n}_consolidation",
                "policy": orch.valuation_policy.to_dict(),
            })
        new_rej = orch.fact_ledger.admission_rejections[rejections_seen:]
        rejections_seen = len(orch.fact_ledger.admission_rejections)
        for r in new_rej:
            rec = dict(r)
            rec["session"] = n
            result["admission_rejections"].append(rec)

        snapshot_ledger(orch, f"session_{n}_consolidation", n,
                        result["ledger_snapshots"], result["vwr_trajectory"],
                        labels, shift_session, stream)

        if n in sleep_after:
            cycle_num += 1
            run_sleep_cycle(orch, cycle_num, n, corpus, cycle_window, stream,
                            labels, shift_session, result, assertions,
                            cell_a_facts)
            cycle_window = []

    result["wall_times"]["wake_and_sleeps"] = round(time.time() - wake_t0, 1)
    if not quick:
        hard_assert(assertions, "total_delivered_events",
                    expected_cum == stream["header"]["total_events"],
                    f"({expected_cum}=={stream['header']['total_events']})")

    # §7.2.4 — final battery
    result["final_battery"] = run_final_battery(orch, corpus, stream, quick,
                                                result)
    result["per_fact_outcomes"] = per_fact_outcomes(orch, stream)

    # C5 self-model update log (full records)
    sm_path = Path(config._data.get("ego", {}).get("self_model_path", ""))
    log_path = sm_path.parent / "self_model_updates.jsonl"
    if log_path.exists():
        with open(log_path) as f:
            result["self_model_updates"] = [json.loads(line)
                                            for line in f if line.strip()]

    result["wall_times"]["total"] = round(time.time() - total_t0, 1)

    # §7.2.5 — delete the run's fused dirs (disk hygiene)
    fused = Path(config.paths["fused_models"])
    if fused.exists():
        shutil.rmtree(fused)
        print(f"[Matrix] Deleted fused dir {fused}")

    return result


# ── §7.3(a) parse-rate pilot ──

def pilot_parse_rate(stream_path, seed=41, n_sessions=PILOT_SESSIONS):
    """Pre-matrix parse-rate check across the four judge conditions (§7.3a).

    Runs each judge policy's score_batch over the first n_sessions of the
    frozen stream (~25 calls total) and exits nonzero if any condition's
    fallback rate exceeds 20% or any pairwise judge-condition rate differs
    by more than 10 pp.
    """
    from src.wake.valuation import build_policy

    stream = load_stream(stream_path)
    sessions = stream["sessions"][:n_sessions]
    print(f"[Pilot] Parse-rate check (§7.3a): {len(sessions)} session(s), "
          f"conditions {JUDGE_CONDITIONS}")

    config, run_dir = resolve_config("judge", seed)
    clean_run_dirs(config, run_dir)
    orch = fresh_orchestrator(config)
    assert_greedy_determinism(orch.backend, {}, "pilot")

    rates = {}
    for cond in JUDGE_CONDITIONS:
        cond_config, _ = resolve_config(cond, seed)
        policy = build_policy(cond_config, orch.backend)
        if policy is None or getattr(policy, "name", None) != cond:
            print(f"[Pilot] FAILED to build policy for '{cond}'")
            sys.exit(1)
        for sess in sessions:
            qa_pairs = [QAPair.from_dict(ev["qa"]) for ev in sess["events"]]
            metas = [ev["meta"] for ev in sess["events"]]
            if qa_pairs:
                policy.score_batch(qa_pairs, metas, [])
        rate = (policy.fallback_count / policy.scored_count
                if policy.scored_count else 0.0)
        rates[cond] = rate
        print(f"[Pilot] {cond}: {policy.call_count} call(s), "
              f"{policy.fallback_count}/{policy.scored_count} "
              f"fallback line(s) ({rate:.1%})")

    breaches = []
    for cond, rate in rates.items():
        if rate > PILOT_MAX_FALLBACK:
            breaches.append(f"{cond} fallback {rate:.1%} > "
                            f"{PILOT_MAX_FALLBACK:.0%}")
    conds = list(rates)
    for i in range(len(conds)):
        for j in range(i + 1, len(conds)):
            diff = abs(rates[conds[i]] - rates[conds[j]])
            if diff > PILOT_MAX_PAIRWISE_DIFF:
                breaches.append(f"|{conds[i]} - {conds[j]}| = "
                                f"{diff * 100:.1f} pp > "
                                f"{PILOT_MAX_PAIRWISE_DIFF * 100:.0f} pp")
    if breaches:
        print("[Pilot] PARSE-RATE PILOT FAILED (§7.3a) — repair prompts "
              "before running the matrix:")
        for b in breaches:
            print(f"  - {b}")
        sys.exit(1)
    print("[Pilot] Parse-rate pilot PASSED (§7.3a)")


# ── Analyze-only ──

def analyze_only(result_path):
    """Print a summary of an existing result JSON. No model."""
    with open(result_path) as f:
        result = json.load(f)
    print(f"=== {result['condition']} seed {result['seed']} ===")
    print("\nVWR trajectory:")
    for v in result.get("vwr_trajectory", []):
        print(f"  {v['after']:<28} vwr={v['vwr']} "
              f"({v['retained_value']}/{v['optimum']}, {v['labels']})")
    fb = result.get("final_battery", {})
    for key in ("future_task", "weights_recall", "contradiction"):
        if key in fb and fb[key]:
            print(f"{key}: {fb[key].get('score')}")
    if fb.get("provenance"):
        print(f"provenance leakage: {fb['provenance'].get('leakage')}")
    print(f"admission rejections: {len(result.get('admission_rejections', []))}")
    bad = [k for k, v in result.get("harness_assertions", {}).items() if not v]
    print(f"harness assertions: "
          f"{'ALL OK' if not bad else 'FAILED: ' + ', '.join(bad)}")
    print(f"wall times: {result.get('wall_times')}")


# ── Main ──

def main():
    # §8 step 14 passthrough subcommands: --freeze / --analyze delegate the
    # remaining args to ego_freeze.py / analyze_ego.py so the pre-registered
    # CLI surface lives on one entry point.
    argv = sys.argv[1:]
    for flag, script in (("--freeze", "ego_freeze.py"),
                         ("--analyze", "analyze_ego.py")):
        if flag in argv:
            import subprocess
            rest = [a for a in argv if a != flag]
            target = REPO_ROOT / "experiments" / script
            print(f"[Matrix] {flag}: delegating to {script} {' '.join(rest)}")
            sys.exit(subprocess.call([sys.executable, str(target)] + rest))

    parser = argparse.ArgumentParser(
        description="EGO-SELECT matrix harness (notes/131 §7.2)")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--quick", action="store_true",
                        help="Pilot: 3 sessions, 1 sleep, reduced battery")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Summarize the existing result JSON (no model)")
    parser.add_argument("--pilot-parse-rate", action="store_true",
                        help="§7.3(a): run the four judge policies over the "
                             "first 3 frozen sessions; exit nonzero if "
                             "fallback > 20%% or pairwise diff > 10 pp")
    parser.add_argument("--freeze", action="store_true",
                        help="Delegate remaining args to ego_freeze.py")
    parser.add_argument("--analyze", action="store_true",
                        help="Delegate remaining args to analyze_ego.py")
    parser.add_argument("--stream", default=str(DEFAULT_STREAM))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.pilot_parse_rate:
        pilot_parse_rate(args.stream)
        return

    if args.condition is None or args.seed is None:
        parser.error("--condition and --seed are required (unless using "
                     "--freeze / --analyze / --pilot-parse-rate)")

    suffix = "_quick" if args.quick else ""
    out_path = Path(args.output) if args.output else (
        RESULTS_DIR / f"ego_{args.condition}_s{args.seed}{suffix}.json")

    if args.analyze_only:
        analyze_only(out_path)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = None
    try:
        result = run_cell(args.condition, args.seed, args.stream,
                          args.corpus, args.quick)
    finally:
        if result is not None:
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n[Matrix] Results saved to {out_path}")

    analyze_only(out_path)


if __name__ == "__main__":
    main()
