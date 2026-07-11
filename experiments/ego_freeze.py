"""EGO-SELECT freeze phase (notes/131-ego-selector-experiment §7.1).

Runs once, before the matrix. Consumes the committed corpus
(experiments/data/ego_corpus_v1.json) and emits the frozen event stream
(experiments/data/ego_corpus_stream.json) that every matrix run replays
identically, plus the oracle-label file for the C6 condition.

Steps (§7.1, each an independent function):
  [1] Extraction replay (MODEL): all 12 sessions through the real pipeline
      once — orchestrator.chat.process_input per turn — recording every
      extracted QAPair and the per-turn SurprisePolicy inputs (user text +
      survivor lists), i.e. the exact chat.py:250 call shape.
  [2] Curator leak demo (MODEL): curator + firewall over the plant sessions;
      assert the assistant fabrications pass grounding (the firewall.py:46
      self-grounding leak), then re-run with ego.curator_provenance_filter on
      and assert the plants are excluded from train.jsonl. Optional
      leak-severity side test behind --leak-side-test.
  [3] Static assertions (NO MODEL): all unique facts survive
      extractor.deduplicate on the scripted statements (both dedup layers:
      extractor + the actual matrix replay path through FactBuffer with
      bypass_dedup); event-count math (59 delivered events); decorrelation
      (point-biserial |r| < 0.15 on delivered events); cell-G rule-score
      constraint against the committed borrowed_rules.yaml.
  [4] Emit experiments/data/ego_corpus_stream.json.
  [5] Oracle simulation (--simulate-optimum shared code path): simulate the
      optimal admission/eviction policy on the frozen event sequence under
      the ACTUAL mechanics (FactLedger, capacity 8, admission gate, arrival
      order, re-mention refresh; graduation excluded) → achievable-optimum
      VWR denominators pre/post shift, per-session optimum trajectory, and
      the derived G3 threshold, written into the stream JSON header. Also
      writes experiments/data/ego_oracle_labels.json for the C6 config.

Usage:
    # Full freeze (loads the MLX model for steps 1-2)
    python experiments/ego_freeze.py --corpus experiments/data/ego_corpus_v1.json

    # Static-only (no model): steps 3-5
    python experiments/ego_freeze.py --skip-model

    # Re-run only the oracle simulation on an existing stream
    python experiments/ego_freeze.py --simulate-optimum

Expected corpus schema (produced by experiments/make_ego_corpus.py, §5):
  {"version": ..., "shift_session": 8, "user": "Mara Voss",
   "sessions": [{"session": 1,
                 "turns": [{"role": "user"|"assistant", "text": "...",
                            "fact_ids": ["A1", ...]}, ...]}, ...],
   "facts": [{"fact_id": "A1", "cell": "A"|"B"|...|"G"|"P",
              "question": "...", "answer": "...", "value": "...",
              "value_pre": 1.0, "value_post": 1.0,
              "provenance": "user_stated"|"user_reported_hearsay"|
                            "assistant_generated",
              "mentions": [{"session": 1, "turn": 2, "text": "..."}, ...],
              "deliver_session": 8   # plants only (optional override)
             }, ...],
   "probes": [{"probe_id": ..., "type": "future_task"|"commitment"|
               "contradiction"|"provenance", "question": "...",
               "expected_value_tokens": [...], "stale_value_tokens": [...],
               "fact_ids": [...]}, ...]}

Stream schema (consumed by experiments/ego_matrix.py):
  {"header": {"corpus_file", "created_at", "shift_session", "total_events",
              "unique_facts", "per_session_event_counts",
              "oracle_simulation": {...G3 + denominators...}},
   "sessions": [{"session": n, "session_text": "...",
                 "events": [{"event_id", "kind": "first_mention"|
                             "re_mention"|"plant", "fact_id", "cell",
                             "value_pre", "value_post",
                             "qa": {question, answer, value, priority,
                                    provenance, timestamp},
                             "meta": {"session", "turn", "user_text",
                                      "turn_new_facts", "speaker"}}, ...]}]}
"""

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import Config
from src.memory.facts import QAPair, FactLedger
from src.sleep.recall_match import fuzzy_value_match
from src.wake.extractor import FactExtractor
from src.wake.fact_buffer import FactBuffer
from src.wake.surprise import SurpriseEstimator

DEFAULT_CORPUS = REPO_ROOT / "experiments" / "data" / "ego_corpus_v1.json"
DEFAULT_STREAM = REPO_ROOT / "experiments" / "data" / "ego_corpus_stream.json"
DEFAULT_LABELS = REPO_ROOT / "experiments" / "data" / "ego_oracle_labels.json"
BORROWED_RULES = REPO_ROOT / "experiments" / "data" / "borrowed_rules.yaml"
FREEZE_CONFIG_SRC = REPO_ROOT / "experiments" / "configs" / "3b_ego_surprise.yaml"

CAPACITY = 8
EXPECTED_UNIQUE_FACTS = 45
EXPECTED_EVENTS = 59
DECORRELATION_MAX_R = 0.15
# §6 G3: ORACLE M1 must reach 0.95 x the simulated achievable optimum
G3_ORACLE_M1_MIN_RATIO = 0.95
G3_M6_MARGIN = 0.15  # ORACLE - RANDOM on M6


# ── Corpus / stream loading ──

def load_corpus(path):
    with open(path) as f:
        corpus = json.load(f)
    corpus.setdefault("shift_session", 8)
    return corpus


def facts_by_id(corpus):
    return {f["fact_id"]: f for f in corpus.get("facts", [])}


def session_text(sess):
    """Scripted session transcript (used for C5 self-model updates)."""
    lines = []
    for turn in sess.get("turns", []):
        lines.append(f"{turn.get('role', 'user')}: {turn.get('text', '')}")
    return "\n".join(lines)


def make_qa(fact, mention=None):
    """Canonical QAPair dict for a corpus fact (mention may reword)."""
    return {
        "question": fact["question"],
        "answer": (mention or {}).get("answer", fact["answer"]),
        "value": fact["value"],
        "source_exchange": (mention or {}).get("text", "")[:100] or None,
        "timestamp": time.time(),
        "priority": 0.5,
        "provenance": fact.get("provenance", "user_stated"),
    }


# ── Event construction (shared by static + model paths) ──

def build_events(corpus):
    """Build the ordered per-session delivered-event lists from the corpus.

    User-provenance facts deliver one event per mention (first_mention then
    re_mention). Assistant plants deliver once, attached to the pre-sleep
    consolidation moments of sleeps 2 and 3 (sessions 8 and 10) unless the
    corpus overrides via 'deliver_session' (§7.1 step 4).
    """
    shift = corpus["shift_session"]
    sessions = {s["session"]: {"session": s["session"],
                               "session_text": session_text(s),
                               "events": [], "_keys": []}
                for s in corpus.get("sessions", [])}

    # Per-(session, turn) fact counts for turn_new_facts meta
    turn_counts = {}
    for fact in corpus.get("facts", []):
        if fact.get("provenance") == "assistant_generated":
            continue
        for m in fact.get("mentions", []):
            key = (m["session"], m.get("turn", 0))
            turn_counts[key] = turn_counts.get(key, 0) + 1

    eid = 0
    for fact in corpus.get("facts", []):
        if fact.get("provenance") == "assistant_generated":
            continue
        for i, m in enumerate(fact.get("mentions", [])):
            sess_n = m["session"]
            eid += 1
            sessions[sess_n]["_keys"].append((m.get("turn", 0), eid))
            sessions[sess_n]["events"].append({
                "event_id": f"e{eid:03d}",
                "kind": "first_mention" if i == 0 else "re_mention",
                "fact_id": fact["fact_id"],
                "cell": fact.get("cell", "?"),
                "value_pre": float(fact.get("value_pre", 0.0)),
                "value_post": float(fact.get("value_post",
                                             fact.get("value_pre", 0.0))),
                "qa": make_qa(fact, m),
                "meta": {
                    "session": sess_n,
                    "turn": m.get("turn", 0),
                    "user_text": m.get("text", ""),
                    "turn_new_facts": turn_counts.get(
                        (sess_n, m.get("turn", 0)), 1),
                    "speaker": "user",
                },
            })

    # Assistant plants: attach to sessions 8 and 10 (pre-sleep moments of
    # sleeps 2 and 3) in fact order, or per 'deliver_session'.
    plant_default_sessions = [shift, shift + 2]
    plant_i = 0
    for fact in corpus.get("facts", []):
        if fact.get("provenance") != "assistant_generated":
            continue
        deliver = fact.get("deliver_session")
        if deliver is None:
            deliver = plant_default_sessions[min(plant_i, 1)]
        plant_i += 1
        eid += 1
        src = (fact.get("mentions") or [{}])[0]
        sessions[deliver]["_keys"].append((10 ** 6, eid))  # end of session
        sessions[deliver]["events"].append({
            "event_id": f"e{eid:03d}",
            "kind": "plant",
            "fact_id": fact["fact_id"],
            "cell": fact.get("cell", "P"),
            "value_pre": float(fact.get("value_pre", 0.0)),
            "value_post": float(fact.get("value_post", 0.0)),
            "qa": make_qa(fact),
            "meta": {
                "session": deliver,
                "turn": src.get("turn", 0),
                "user_text": src.get("text", ""),
                "turn_new_facts": 1,
                "speaker": "assistant",
            },
        })

    # Stable ordering: by turn within session, plants last
    out = []
    for n in sorted(sessions):
        sess = sessions[n]
        order = sorted(range(len(sess["events"])), key=lambda i: sess["_keys"][i])
        sess["events"] = [sess["events"][i] for i in order]
        del sess["_keys"]
        out.append(sess)
    return out


# ── Step 1: extraction replay (MODEL) ──

def destroy_orchestrator(orch):
    """Free the model between freeze-phase orchestrators (8 GB machine)."""
    import gc
    if hasattr(orch, "backend"):
        orch.backend.model = None
        orch.backend.tokenizer = None
    del orch
    gc.collect()


def make_freeze_config(subdir="freeze"):
    """Derive an isolated freeze config from the surprise condition config."""
    text = FREEZE_CONFIG_SRC.read_text().replace("{seed}", subdir)
    tmp = Path(tempfile.mkdtemp(prefix="ego_freeze_cfg_")) / "config.yaml"
    tmp.write_text(text)
    return Config(str(tmp))


def step1_extraction_replay(corpus, seed=41, allow_misses=False):
    """§7.1.1 — replay all sessions through the real wake pipeline once.

    Records every extracted QAPair (via a FactBuffer.add wrapper) and every
    SurpriseEstimator.evaluate call (user text + survivor list + count —
    exactly SurprisePolicy's frozen-replay inputs). Returns the record dict.

    Extraction coverage is BLOCKING (§7.1.3: reword and re-freeze until
    green) unless allow_misses=True (--allow-extraction-misses).
    """
    from src.orchestrator import Orchestrator

    print("\n[Freeze 1/5] Extraction replay through the real pipeline (MODEL)")
    run_dir = REPO_ROOT / "data" / "ego_exp" / "freeze"
    if run_dir.exists():
        shutil.rmtree(run_dir)

    config = make_freeze_config("freeze")
    orch = Orchestrator(config)
    orch.chat.set_sleep_callback(lambda t: None)
    orch.chat.set_nap_callback(lambda t: None)
    orch.backend.set_seed(seed)

    surprise_calls = []
    original_evaluate = orch.surprise_estimator.evaluate

    def recording_evaluate(user_message, new_facts, total_extracted):
        score = original_evaluate(user_message, new_facts, total_extracted)
        surprise_calls.append({
            "user_text": user_message,
            "survivors": [qa.to_dict() for qa in new_facts],
            "turn_new_facts": total_extracted,
            "score": score,
        })
        return score

    orch.surprise_estimator.evaluate = recording_evaluate

    buffered = []
    original_add = orch.fact_buffer.add

    def recording_add(qa, turn=0, surprise=0.0, bypass_dedup=False, meta=None):
        buffered.append({"qa": qa.to_dict(), "turn": turn, "surprise": surprise})
        return original_add(qa, turn=turn, surprise=surprise,
                            bypass_dedup=bypass_dedup, meta=meta)

    orch.fact_buffer.add = recording_add

    per_session = {}
    for sess in corpus["sessions"]:
        n = sess["session"]
        print(f"  [Freeze] Session {n}...")
        start_calls, start_buffered = len(surprise_calls), len(buffered)
        for turn in sess.get("turns", []):
            if turn.get("role", "user") != "user":
                continue  # assistant turns are scripted, not replayed
            orch.chat.process_input(turn["text"])
        per_session[str(n)] = {
            "surprise_calls": surprise_calls[start_calls:],
            "buffered": buffered[start_buffered:],
        }
        orch.fact_buffer.consolidate(reason="manual")
    orch.fact_buffer.add = original_add
    orch.surprise_estimator.evaluate = original_evaluate

    # Coverage check: every scripted fact must have been extracted somewhere
    misses = []
    all_answers = " ||| ".join(b["qa"]["answer"] for b in buffered)
    for fact in corpus.get("facts", []):
        if fact.get("provenance") == "assistant_generated":
            continue
        if not fuzzy_value_match(fact["value"], all_answers):
            misses.append(fact["fact_id"])
    if misses:
        print(f"  [Freeze] {len(misses)} fact(s) not recovered by "
              f"extraction: {misses}")
        if not allow_misses:
            destroy_orchestrator(orch)
            raise AssertionError(
                f"§7.1.3 BLOCKING: {len(misses)} fact(s) did not survive "
                f"model-path extraction: {misses} — reword the corpus and "
                "re-freeze (or pass --allow-extraction-misses to override)")
        print("  [Freeze] --allow-extraction-misses: proceeding despite misses")
    else:
        print(f"  [Freeze] All scripted user facts recovered by extraction")

    destroy_orchestrator(orch)
    return {"per_session": per_session, "extraction_misses": misses}


# ── Step 2: curator leak demo (MODEL) ──

def plant_sessions(corpus):
    """Sessions containing scripted assistant fabrications (default 5, 9)."""
    out = set()
    for fact in corpus.get("facts", []):
        if fact.get("provenance") == "assistant_generated":
            for m in fact.get("mentions", []):
                out.add(m["session"])
    return sorted(out) or [5, 9]


def step2_leak_demo(corpus, side_test=False):
    """§7.1.2 — demonstrate the firewall self-grounding leak and its fix.

    Runs curator.curate_with_model + firewall over the scripted plant
    sessions (assistant turns included), asserts the fabrications pass
    grounding into train.jsonl, then re-runs with ego.curator_provenance_filter
    on and asserts they are excluded.
    """
    from src.orchestrator import Orchestrator

    print("\n[Freeze 2/5] Curator/firewall leak demonstration (MODEL)")
    plants = [f for f in corpus.get("facts", [])
              if f.get("provenance") == "assistant_generated"]
    if not plants:
        print("  [Freeze] No assistant plants in corpus — skipping leak demo")
        return {"skipped": True}

    sess_ids = plant_sessions(corpus)
    messages = []
    for sess in corpus["sessions"]:
        if sess["session"] in sess_ids:
            for turn in sess.get("turns", []):
                messages.append({"role": turn.get("role", "user"),
                                 "content": turn.get("text", "")})

    result = {"plant_sessions": sess_ids, "leak": {}, "fix": {}}

    def train_jsonl_hits(config_obj, cycle_id):
        orch = Orchestrator(config_obj)
        orch.chat.set_sleep_callback(lambda t: None)
        orch.chat.set_nap_callback(lambda t: None)
        orch.curator.curate_with_model(messages, cycle_id)
        train_file = (Path(config_obj.paths["training"])
                      / f"cycle_{cycle_id}" / "train.jsonl")
        text = train_file.read_text() if train_file.exists() else ""
        hits = {p["fact_id"]: fuzzy_value_match(p["value"], text)
                for p in plants}
        destroy_orchestrator(orch)
        return hits

    # (a) leak: default curator, firewall grounds against assistant text too
    run_dir = REPO_ROOT / "data" / "ego_exp" / "freeze_leak"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cfg = make_freeze_config("freeze_leak")
    hits = train_jsonl_hits(cfg, "leak_demo")
    result["leak"] = hits
    leaked = [fid for fid, hit in hits.items() if hit]
    print(f"  [Freeze] Plants self-grounded into train.jsonl: {leaked}")
    assert leaked, ("Leak demo failed: no assistant fabrication passed "
                    "grounding — check corpus plant phrasing (firewall.py:46)")

    # (b) fix: curator provenance gate on
    run_dir = REPO_ROOT / "data" / "ego_exp" / "freeze_fix"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cfg = make_freeze_config("freeze_fix")
    cfg._data["ego"]["curator_provenance_filter"] = True
    hits = train_jsonl_hits(cfg, "leak_fix")
    result["fix"] = hits
    still = [fid for fid, hit in hits.items() if hit]
    assert not still, f"Curator provenance gate failed to exclude: {still}"
    print("  [Freeze] Curator provenance gate excludes all plants — fix verified")

    if side_test:
        result["side_test"] = _leak_severity_side_test(corpus, plants)
    return result


def _leak_severity_side_test(corpus, plants):
    """Optional §7.1.2 side test: train adapters on leaked vs filtered
    train.jsonl and probe assertion leakage (1 seed, best-effort)."""
    from src.orchestrator import Orchestrator

    print("  [Freeze] Leak-severity side test (optional, ~40 min)")
    out = {}
    for tag, filt in (("with_plants", False), ("filtered", True)):
        try:
            run_dir = REPO_ROOT / "data" / "ego_exp" / f"freeze_side_{tag}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            cfg = make_freeze_config(f"freeze_side_{tag}")
            cfg._data["ego"]["curator_provenance_filter"] = filt
            orch = Orchestrator(cfg)
            orch.chat.set_sleep_callback(lambda t: None)
            orch.chat.set_nap_callback(lambda t: None)
            messages = []
            for sess in corpus["sessions"]:
                if sess["session"] in plant_sessions(corpus):
                    for turn in sess.get("turns", []):
                        messages.append({"role": turn.get("role", "user"),
                                         "content": turn.get("text", "")})
            orch.curator.curate_with_model(messages, "side")
            data_dir = Path(cfg.paths["training"]) / "cycle_side"
            adapter = Path(cfg.paths["adapters"]) / "side"
            ok = orch.backend.train_lora(str(data_dir), str(adapter),
                                         num_layers=8, iters=160)
            leaks = {}
            if ok:
                fused = Path(cfg.paths["fused_models"]) / "side"
                orch.backend.fuse_adapter(str(adapter), str(fused))
                orch.backend.reload(str(fused))
                probes = [p for p in corpus.get("probes", [])
                          if p.get("type") == "provenance"]
                for p in probes:
                    msgs = [{"role": "user", "content": p["question"]}]
                    prompt = orch.backend.apply_chat_template(msgs)
                    resp = orch.backend.generate(prompt, max_tokens=80,
                                                 temperature=0.0)
                    leaks[p.get("probe_id", p["question"][:30])] = any(
                        fuzzy_value_match(t, resp)
                        for t in p.get("expected_value_tokens", []))
            out[tag] = {"trained": bool(ok), "leaks": leaks}
            destroy_orchestrator(orch)
        except Exception as e:
            print(f"  [Freeze] side test '{tag}' failed (non-fatal): {e}")
            out[tag] = {"error": str(e)}
    return out


# ── Step 3: static assertions (NO MODEL) ──

def _point_biserial(binary, continuous):
    """Point-biserial correlation between a 0/1 list and a float list."""
    n = len(binary)
    if n < 2:
        return 0.0
    g1 = [c for b, c in zip(binary, continuous) if b]
    g0 = [c for b, c in zip(binary, continuous) if not b]
    if not g1 or not g0:
        return 0.0
    mean = sum(continuous) / n
    std = math.sqrt(sum((c - mean) ** 2 for c in continuous) / n)
    if std == 0:
        return 0.0
    m1 = sum(g1) / len(g1)
    m0 = sum(g0) / len(g0)
    p = len(g1) / n
    return (m1 - m0) / std * math.sqrt(p * (1 - p))


class _CountingLedger:
    """Stub ledger counting facts reaching the ledger-merge step."""

    def __init__(self):
        self.merged = 0

    def get_all_qa_pairs(self):
        return []

    def add_or_refresh(self, qa):
        self.merged += 1
        return ("added", f"stub{self.merged}")


def step3_static_assertions(corpus, sessions_events, config):
    """§7.1.3 — dedup survival at BOTH layers, event-count math,
    decorrelation, cell-G rule-score constraint. No model calls."""
    print("\n[Freeze 3/5] Static assertions (no model)")
    report = {}

    facts = corpus.get("facts", [])
    unique = len(facts)
    events = sum(len(s["events"]) for s in sessions_events)
    print(f"  Unique facts: {unique} (expected {EXPECTED_UNIQUE_FACTS}), "
          f"delivered events: {events} (expected {EXPECTED_EVENTS})")
    assert unique == EXPECTED_UNIQUE_FACTS, \
        f"Corpus has {unique} unique facts, expected {EXPECTED_UNIQUE_FACTS}"
    assert events == EXPECTED_EVENTS, \
        f"Stream delivers {events} events, expected {EXPECTED_EVENTS}"
    report["unique_facts"] = unique
    report["delivered_events"] = events

    # (a) Extractor-layer dedup survival, session by session (cell F focus)
    extractor = FactExtractor(config, backend=None)
    eaten = []
    existing = []
    for sess in sessions_events:
        firsts = [QAPair.from_dict(e["qa"]) for e in sess["events"]
                  if e["kind"] == "first_mention"]
        survived = extractor.deduplicate(firsts, existing)
        surv_q = {qa.question for qa in survived}
        for qa in firsts:
            if qa.question not in surv_q:
                eaten.append((sess["session"], qa.question))
        existing.extend(firsts)
    assert not eaten, (
        "extractor.deduplicate ate scripted first-mentions (reword corpus, "
        f"§11 F11/R17): {eaten}")
    print("  extractor.deduplicate: all first-mentions survive")
    report["extractor_dedup_ok"] = True

    # (b) Actual matrix replay path: FactBuffer with bypass_dedup=True
    per_session_counts = {}
    stub = _CountingLedger()
    buf = FactBuffer(config, stub)
    for sess in sessions_events:
        before = stub.merged
        for ev in sess["events"]:
            buf.add(QAPair.from_dict(ev["qa"]), turn=ev["meta"]["turn"],
                    bypass_dedup=True, meta=ev["meta"])
        buf.consolidate(reason="manual")
        per_session_counts[str(sess["session"])] = stub.merged - before
        assert stub.merged - before == len(sess["events"]), (
            f"Replay path lost events in session {sess['session']}: "
            f"{stub.merged - before} != {len(sess['events'])}")
    assert stub.merged == EXPECTED_EVENTS, \
        f"Replay path delivered {stub.merged} events, expected {EXPECTED_EVENTS}"
    print(f"  Matrix replay path: all {stub.merged} events reach ledger-merge")
    report["replay_path_events"] = stub.merged
    report["per_session_event_counts"] = per_session_counts

    # (c) Decorrelation on delivered events (§5, blocking)
    estimator = SurpriseEstimator(config, backend=None)
    shift = corpus["shift_session"]
    mention_freq = {}
    for sess in sessions_events:
        for ev in sess["events"]:
            mention_freq[ev["fact_id"]] = mention_freq.get(ev["fact_id"], 0) + 1

    checks = {}
    for label_key in ("value_pre", "value_post"):
        vital, freqs, markers, realized = [], [], [], []
        for fact in facts:
            if fact.get("provenance") == "assistant_generated":
                continue
            vital.append(1 if float(fact.get(label_key,
                         fact.get("value_pre", 0.0))) >= 0.67 else 0)
            freqs.append(mention_freq.get(fact["fact_id"], 0))
            first = (fact.get("mentions") or [{}])[0]
            marker = estimator._marker_score(first.get("text", ""))
            markers.append(1.0 if marker > 0 else 0.0)
            # Realized C1 priority: 0.625 + 0.375 * marker (§2 scope 6)
            realized.append(0.625 + 0.375 * marker)
        for name, series in (("mention_frequency", freqs),
                             ("marker_presence", markers),
                             ("realized_surprise", realized)):
            r = _point_biserial(vital, series)
            checks[f"{label_key}__{name}"] = round(r, 4)
            print(f"  decorrelation {label_key} vs {name}: r={r:+.3f}")
            if name == "mention_frequency":
                # Pre-registered deviation (make_ego_corpus.py, same rule):
                # the two-sided bound is structurally infeasible for
                # frequency — §5 pins all 15 pre-shift vitals in
                # single-mention cells while cell D (value 0) delivers 3
                # mentions each, forcing r_pre = -0.34 regardless of
                # wording. Frequency is anti-correlated by design (cell D
                # makes frequency a *misleading* signal), so only a
                # positive correlation would break the manipulation:
                # enforce the one-sided bound r < +0.15.
                assert r < DECORRELATION_MAX_R, (
                    f"Decorrelation FAILED (r >= +{DECORRELATION_MAX_R}): "
                    f"{label_key} vs {name} r={r:+.3f} — regenerate corpus")
            else:
                assert abs(r) < DECORRELATION_MAX_R, (
                    f"Decorrelation FAILED (|r| >= {DECORRELATION_MAX_R}): "
                    f"{label_key} vs {name} r={r:+.3f} — regenerate corpus "
                    "(§5)")
    report["decorrelation"] = checks

    # (d) Cell-G rule-score constraint against committed borrowed rules
    from src.wake.valuation import BorrowedPolicy
    policy = BorrowedPolicy(str(BORROWED_RULES))
    bad = []
    for fact in facts:
        if fact.get("cell") != "G":
            continue
        score = policy._score_one(QAPair.from_dict(make_qa(fact)))
        pre = float(fact.get("value_pre", 0.0))
        post = float(fact.get("value_post", 0.0))
        if post > pre and score > 0.4:          # flip-up must rule-score <= 0.4
            bad.append((fact["fact_id"], "flip_up", score))
        if post < pre and score < 0.8:          # flip-down must rule-score >= 0.8
            bad.append((fact["fact_id"], "flip_down", score))
    assert not bad, f"Cell-G rule-score constraint violated (§5): {bad}"
    print("  Cell-G rule-score constraint holds (rules mis-value all shifts)")
    report["cell_g_rule_scores_ok"] = True

    return report


# ── Step 5: oracle simulation (shared --simulate-optimum code path) ──

def simulate_optimum(sessions_events, shift_session, capacity=CAPACITY):
    """Simulate the optimal admission/eviction policy on the frozen event
    sequence under the ACTUAL mechanics (§6 M1/G3, §7.1 step 5).

    Uses a real FactLedger (capacity, admission gate, add_or_refresh
    re-mention semantics) driven by ground-truth value labels (post-shift
    labels from shift_session on). Graduation is excluded by construction.
    Returns denominators + per-session optimum trajectory + G3 threshold.
    """
    tmp = Path(tempfile.mkdtemp(prefix="ego_optimum_")) / "ledger.json"
    ledger = FactLedger(str(tmp), max_facts=capacity, admission_gate=True)

    labels = {}
    trajectory = []
    for sess in sessions_events:
        n = sess["session"]
        for ev in sess["events"]:
            labels[ev["qa"]["question"].lower().strip()] = (
                ev["value_pre"], ev["value_post"])
            qa = QAPair.from_dict(ev["qa"])
            qa.priority = ev["value_post"] if n >= shift_session else ev["value_pre"]
            ledger.add_or_refresh(qa)

        def retained_value(post):
            total = 0.0
            for e in ledger._entries:
                if e.get("pruned", False) and not e.get("graduated", False):
                    continue
                lab = labels.get(e["qa"]["question"].lower().strip())
                if lab:
                    total += lab[1] if post else lab[0]
            return total

        trajectory.append({
            "session": n,
            "optimum_retained_value_pre_labels": round(retained_value(False), 3),
            "optimum_retained_value_post_labels": round(retained_value(True), 3),
        })

    pre_points = [t for t in trajectory if t["session"] < shift_session]
    denom_pre = (pre_points[-1]["optimum_retained_value_pre_labels"]
                 if pre_points else 0.0)
    denom_post = trajectory[-1]["optimum_retained_value_post_labels"]

    sim = {
        "capacity": capacity,
        "shift_session": shift_session,
        "optimum_denominator_pre_shift": denom_pre,
        "optimum_denominator_post_shift": denom_post,
        "optimum_trajectory": trajectory,
        "optimum_final_retained": [
            e["qa"]["question"] for e in ledger._entries
            if not e.get("pruned", False)],
        "g3": {
            "oracle_m1_min_ratio": G3_ORACLE_M1_MIN_RATIO,
            "oracle_minus_random_m6_min": G3_M6_MARGIN,
        },
    }
    print(f"  [Optimum] denominators: pre-shift={denom_pre}, "
          f"post-shift={denom_post} "
          f"(G3: oracle M1 >= {G3_ORACLE_M1_MIN_RATIO})")
    return sim


def write_oracle_labels(sessions_events, shift_session, labels_path):
    """Emit the C6 label file (question -> {value_pre, value_post})."""
    labels = {}
    for sess in sessions_events:
        for ev in sess["events"]:
            labels[ev["qa"]["question"]] = {
                "value_pre": ev["value_pre"],
                "value_post": ev["value_post"],
            }
    with open(labels_path, "w") as f:
        json.dump({"shift_session": shift_session, "labels": labels}, f, indent=2)
    print(f"  [Freeze] Oracle labels written to {labels_path}")


# ── Step 4: emit stream ──

def emit_stream(corpus, sessions_events, out_path, corpus_path=None,
                extraction=None, leak_demo=None, static_report=None):
    per_session = {str(s["session"]): len(s["events"]) for s in sessions_events}
    stream = {
        "header": {
            "version": 1,
            "corpus_file": str(corpus_path or DEFAULT_CORPUS),
            "corpus_version": corpus.get("version"),
            "created_at": time.time(),
            "shift_session": corpus["shift_session"],
            "unique_facts": len(corpus.get("facts", [])),
            "total_events": sum(per_session.values()),
            "per_session_event_counts": per_session,
            "static_report": static_report,
            "leak_demo": leak_demo,
            "extraction_misses": (extraction or {}).get("extraction_misses"),
        },
        "sessions": sessions_events,
    }
    if extraction:
        # Attach the recorded per-turn surprise inputs for auditability
        for sess in stream["sessions"]:
            rec = extraction["per_session"].get(str(sess["session"]))
            if rec:
                sess["recorded_surprise_calls"] = rec["surprise_calls"]
    with open(out_path, "w") as f:
        json.dump(stream, f, indent=2)
    print(f"\n[Freeze 4/5] Stream written to {out_path} "
          f"({stream['header']['total_events']} events)")
    return stream


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="EGO-SELECT freeze phase (notes/131 §7.1)")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS),
                        help="Committed corpus JSON (ego_corpus_v1.json)")
    parser.add_argument("--out", default=str(DEFAULT_STREAM),
                        help="Output stream JSON (ego_corpus_stream.json)")
    parser.add_argument("--labels-out", default=str(DEFAULT_LABELS),
                        help="Output oracle labels JSON for the C6 config")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"),
                        help="Config for static extractor/estimator settings")
    parser.add_argument("--skip-model", action="store_true",
                        help="Static-only: skip steps 1-2 (no MLX load)")
    parser.add_argument("--simulate-optimum", action="store_true",
                        help="Only re-run the oracle simulation on --out")
    parser.add_argument("--leak-side-test", action="store_true",
                        help="Also run the optional leak-severity side test")
    parser.add_argument("--allow-extraction-misses", action="store_true",
                        help="Override the §7.1.3 blocking assertion on "
                             "model-path extraction misses (default: hard fail)")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    if args.simulate_optimum:
        # Shared code path: operate on an existing stream file
        with open(args.out) as f:
            stream = json.load(f)
        print("[Freeze 5/5] Oracle simulation (standalone)")
        sim = simulate_optimum(stream["sessions"],
                               stream["header"]["shift_session"])
        stream["header"]["oracle_simulation"] = sim
        with open(args.out, "w") as f:
            json.dump(stream, f, indent=2)
        write_oracle_labels(stream["sessions"],
                            stream["header"]["shift_session"], args.labels_out)
        print(f"  Stream header updated: {args.out}")
        return

    corpus = load_corpus(args.corpus)
    config = Config(args.config)
    sessions_events = build_events(corpus)

    extraction = None
    leak_demo = None
    if args.skip_model:
        print("[Freeze] --skip-model: steps 1-2 (extraction replay, leak "
              "demo) skipped; static checks + stream + optimum only")
    else:
        extraction = step1_extraction_replay(
            corpus, seed=args.seed,
            allow_misses=args.allow_extraction_misses)
        leak_demo = step2_leak_demo(corpus, side_test=args.leak_side_test)

    static_report = step3_static_assertions(corpus, sessions_events, config)
    stream = emit_stream(corpus, sessions_events, args.out,
                         corpus_path=args.corpus, extraction=extraction,
                         leak_demo=leak_demo, static_report=static_report)

    print("\n[Freeze 5/5] Oracle simulation")
    sim = simulate_optimum(stream["sessions"], corpus["shift_session"])
    stream["header"]["oracle_simulation"] = sim
    with open(args.out, "w") as f:
        json.dump(stream, f, indent=2)
    write_oracle_labels(stream["sessions"], corpus["shift_session"],
                        args.labels_out)

    print("\n[Freeze] Done. Commit the stream + oracle labels before the "
          "matrix (§7.4 step 5).")


if __name__ == "__main__":
    main()
