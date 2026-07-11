"""EGO-SELECT analysis (notes/131-ego-selector-experiment §3, §6, §9, §7.4.8).

Consumes the matrix result JSONs (experiments/results/ego_<cond>_s<seed>.json,
schema in ego_matrix.py's docstring) plus the frozen stream
(experiments/data/ego_corpus_stream.json, schema in ego_freeze.py's docstring),
computes the per-run metrics M1-M10 and gates G1-G4, runs the §9 inference
layer (run-level sign-flip permutation within seed, seed sign-consistency
co-criterion, exact McNemar on (fact_id, seed) pairs, cluster bootstrap on
fact_id with 10k seeded draws, Holm correction across the 7 primary contrasts
within endpoint family), evaluates §3's pre-registered decision rules —
encoded as data in PREDICTIONS, not prose — and emits a notes/133-style
markdown skeleton (verdict table, per-cell survival heatmap, rho bars as text,
every-null statistics appendix, Files section) to stdout or --out.

Also provides --simulate-optimum (§7.1 step 5): the achievable-optimum VWR
denominator simulation, reusing ego_freeze.simulate_optimum when importable
and falling back to an identical local reimplementation otherwise.

stdlib only; scipy is optional (used for Spearman when installed, with a
hand-rolled fallback). Missing runs are handled gracefully: a partial matrix
marks the affected contrasts INCONCLUSIVE.

Usage:
    python experiments/analyze_ego.py                       # skeleton -> stdout
    python experiments/analyze_ego.py --out notes/133-draft.md
    python experiments/analyze_ego.py --simulate-optimum    # M1 denominators + G3
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from scipy import stats as _scipy_stats  # optional (§8 step 14)
except ImportError:
    _scipy_stats = None

CONDITIONS = ["random", "surprise", "borrowed", "judge", "judge_rules",
              "ego_static", "ego_full", "oracle"]
COND_LABELS = {"random": "C0 RANDOM", "surprise": "C1 SURPRISE",
               "borrowed": "C2 BORROWED", "judge": "C3 JUDGE",
               "judge_rules": "C3b JUDGE-RULES", "ego_static": "C4 EGO-STATIC",
               "ego_full": "C5 EGO-FULL", "oracle": "C6 ORACLE"}
DEFAULT_SEEDS = [41, 42, 43]
DEFAULT_STREAM = REPO_ROOT / "experiments" / "data" / "ego_corpus_stream.json"
DEFAULT_CORPUS = REPO_ROOT / "experiments" / "data" / "ego_corpus_v1.json"
DEFAULT_LABELS = REPO_ROOT / "experiments" / "data" / "ego_oracle_labels.json"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

ALPHA = 0.05
APPROX_MARGIN = 0.10          # §3: "~=" means |delta| < 0.10 with CI incl. 0
N_BOOT = 10000                # §9 cluster bootstrap draws
N_PERM_POOLED = 10000         # pooled rho permutation draws
RNG_SEED = 20260710
VITAL_THRESHOLD = 0.67        # §5: value >= 0.67 iff required by >= 1 probe
G2_VOID_RATE = 0.30           # §6 G2: > 30% fallback voids the condition
G2_SENSITIVITY_RATE = 0.10    # 10-30% -> parsed-only sensitivity analysis
G2_DIFFERENTIAL_PP = 0.10     # |C4 - C3| > 10 pp flags a confound on P2
G3_M6_MARGIN = 0.15           # §6 G3: ORACLE - RANDOM on M6
G3_M1_MIN = 0.95              # §6 G3: ORACLE M1 >= 0.95 x achievable optimum
P7_MAX_LATENCY_S = 10.0       # §3 P7: < 10 s added latency per moment
M4_REVALUE_FRACTION = 2.0 / 3.0
CAPACITY = 8


# ── Oracle simulation (--simulate-optimum, §7.1 step 5) ──
# Reuse ego_freeze's implementation when importable; the local fallback below
# is an identical reimplementation (same mechanics, same output shape).

def _simulate_optimum_local(sessions_events, shift_session, capacity=CAPACITY):
    """Identical reimplementation of ego_freeze.simulate_optimum."""
    import tempfile
    from src.memory.facts import QAPair, FactLedger

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
            "oracle_m1_min_ratio": G3_M1_MIN,
            "oracle_minus_random_m6_min": G3_M6_MARGIN,
        },
    }
    print(f"  [Optimum] denominators: pre-shift={denom_pre}, "
          f"post-shift={denom_post} (G3: oracle M1 >= {G3_M1_MIN})")
    return sim


def _write_oracle_labels_local(sessions_events, shift_session, labels_path):
    """Identical reimplementation of ego_freeze.write_oracle_labels."""
    labels = {}
    for sess in sessions_events:
        for ev in sess["events"]:
            labels[ev["qa"]["question"]] = {
                "value_pre": ev["value_pre"],
                "value_post": ev["value_post"],
            }
    with open(labels_path, "w") as f:
        json.dump({"shift_session": shift_session, "labels": labels}, f,
                  indent=2)
    print(f"  [Analyze] Oracle labels written to {labels_path}")


def cmd_simulate_optimum(stream_path, labels_path):
    """§7.1 step 5: simulate the achievable optimum on the frozen stream,
    write denominators + G3 threshold into the stream header."""
    try:
        from ego_freeze import simulate_optimum, write_oracle_labels
        print("[Analyze] Using ego_freeze.simulate_optimum (shared code path)")
    except Exception as e:
        print(f"[Analyze] ego_freeze not importable ({e}) — "
              "using the identical local simulation")
        simulate_optimum = _simulate_optimum_local
        write_oracle_labels = _write_oracle_labels_local

    try:
        with open(stream_path) as f:
            stream = json.load(f)
    except OSError as e:
        print(f"[Analyze] Cannot read stream ({e}) — run ego_freeze.py first "
              "(§7.1 step 4)")
        sys.exit(1)
    print("[Analyze] Oracle simulation on the frozen stream")
    sim = simulate_optimum(stream["sessions"],
                           stream["header"]["shift_session"])
    stream["header"]["oracle_simulation"] = sim
    with open(stream_path, "w") as f:
        json.dump(stream, f, indent=2)
    write_oracle_labels(stream["sessions"],
                        stream["header"]["shift_session"], labels_path)
    print(f"[Analyze] Stream header updated: {stream_path}")


# ── stdlib statistics (§9) ──

def _rankdata(xs):
    """Average ranks with ties (1-based)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(xs, ys):
    """Spearman rank correlation; None if undefined (<3 points or no variance)."""
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    if _scipy_stats is not None:
        res = _scipy_stats.spearmanr(xs, ys)
        rho = getattr(res, "statistic", getattr(res, "correlation", None))
        return None if (rho is None or rho != rho) else float(rho)
    rx, ry = _rankdata(xs), _rankdata(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def runlevel_perm_p(observed_diffs, pools, rng, n_mc=20000,
                    max_exact=300000):
    """Run-level permutation p (§9): permute the condition-label assignment
    within each seed and recompute the pairwise contrast. Under the null the
    per-seed diff is the difference between two randomly relabeled runs from
    that seed's pool (all conditions with data); one-sided
    (mean >= observed). Exact enumeration when the product of per-seed
    ordered-pair counts is small, else seeded Monte Carlo."""
    import itertools
    if not observed_diffs or any(len(p) < 2 for p in pools):
        return None
    n = len(observed_diffs)
    observed = sum(observed_diffs) / n
    pair_diffs = []
    for pool in pools:
        pair_diffs.append([x - y for i, x in enumerate(pool)
                           for j, y in enumerate(pool) if i != j])
    total = 1
    for pd in pair_diffs:
        total *= len(pd)
    if total <= max_exact:
        count = sum(1 for combo in itertools.product(*pair_diffs)
                    if sum(combo) / n >= observed - 1e-12)
        return count / total
    count = 0
    for _ in range(n_mc):
        s = sum(rng.choice(pd) for pd in pair_diffs)
        if s / n >= observed - 1e-12:
            count += 1
    return (count + 1) / (n_mc + 1)


def pooled_perm_p(pairs_a, pairs_b, n_perm, rng):
    """Permutation p for a rho contrast on pooled (score, value) pairs (§9):
    permute the condition assignment, one-sided (rho_a - rho_b)."""
    ra = spearman_rho([p[0] for p in pairs_a], [p[1] for p in pairs_a])
    rb = spearman_rho([p[0] for p in pairs_b], [p[1] for p in pairs_b])
    if ra is None or rb is None:
        return None, ra, rb
    observed = ra - rb
    pooled = list(pairs_a) + list(pairs_b)
    na = len(pairs_a)
    count = 0
    valid = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        pa = spearman_rho([p[0] for p in pooled[:na]],
                          [p[1] for p in pooled[:na]])
        pb = spearman_rho([p[0] for p in pooled[na:]],
                          [p[1] for p in pooled[na:]])
        if pa is None or pb is None:
            continue
        valid += 1
        if pa - pb >= observed - 1e-12:
            count += 1
    if valid == 0:
        return None, ra, rb
    return (count + 1) / (valid + 1), ra, rb


def mcnemar_exact_p(b, c):
    """Exact two-sided McNemar on discordant pair counts (§9, secondary)."""
    n = b + c
    if n == 0:
        return 1.0
    if _scipy_stats is not None:
        return float(min(1.0, 2 * _scipy_stats.binom.cdf(min(b, c), n, 0.5)))
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2.0 ** n
    return min(1.0, 2 * tail)


def cluster_bootstrap_ci(diffs_by_item, n_boot, rng, denom=None):
    """Cluster bootstrap on fact_id (§9): resample items with replacement,
    stat = sum(diffs)/denom (denom None -> mean). Returns (est, lo, hi)."""
    items = sorted(diffs_by_item)
    if not items:
        return None
    vals = [diffs_by_item[i] for i in items]
    n = len(vals)

    def stat(sample):
        s = sum(sample)
        return s / denom if denom else s / len(sample)

    est = stat(vals)
    draws = []
    for _ in range(n_boot):
        draws.append(stat([vals[rng.randrange(n)] for _ in range(n)]))
    draws.sort()
    lo = draws[int(0.025 * n_boot)]
    hi = draws[min(int(0.975 * n_boot), n_boot - 1)]
    return est, lo, hi


def holm_adjust(pvals):
    """Holm step-down: {key: p} -> {key: adjusted p}."""
    items = sorted(((k, p) for k, p in pvals.items() if p is not None),
                   key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        adjusted[key] = running
    for k, p in pvals.items():
        if p is None:
            adjusted[k] = None
    return adjusted


# ── Context: stream, labels, rule coverage ──

def _qkey(question):
    return question.lower().strip()


def load_context(stream_path, corpus_path):
    """Load the frozen stream + corpus and build the fact tables."""
    try:
        with open(stream_path) as f:
            stream = json.load(f)
    except OSError as e:
        print(f"[Analyze] WARNING: stream unavailable ({e}) — "
              "metrics needing labels will be None")
        stream = {"header": {"shift_session": 8}, "sessions": []}
    try:
        with open(corpus_path) as f:
            corpus = json.load(f)
    except OSError:
        corpus = {}

    facts = {}
    for sess in stream.get("sessions", []):
        for ev in sess["events"]:
            fid = ev["fact_id"]
            if fid not in facts:
                facts[fid] = {
                    "fact_id": fid,
                    "cell": ev["cell"],
                    "value_pre": float(ev["value_pre"]),
                    "value_post": float(ev["value_post"]),
                    "question": ev["qa"]["question"],
                    "qkey": _qkey(ev["qa"]["question"]),
                    "answer": ev["qa"].get("answer", ""),
                    "provenance": ev["qa"].get("provenance", "user_stated"),
                }

    sim = stream.get("header", {}).get("oracle_simulation") or {}
    ctx = {
        "stream": stream,
        "corpus": corpus,
        "facts": facts,
        "by_qkey": {f["qkey"]: f for f in facts.values()},
        "shift_session": stream.get("header", {}).get("shift_session", 8),
        "denom_pre": sim.get("optimum_denominator_pre_shift"),
        "denom_post": sim.get("optimum_denominator_post_shift"),
        "g3": sim.get("g3", {"oracle_m1_min_ratio": G3_M1_MIN,
                             "oracle_minus_random_m6_min": G3_M6_MARGIN}),
        "rule_uncovered": rule_uncovered_facts(facts),
    }
    return ctx


def rule_uncovered_facts(facts):
    """fact_ids matching NO borrowed rule (score == default) — the P2b/P3
    rule-uncovered subset. None when the rules cannot be scored."""
    try:
        from src.wake.valuation import BorrowedPolicy
        from src.memory.facts import QAPair
        policy = BorrowedPolicy()
    except Exception as e:
        print(f"[Analyze] WARNING: cannot score borrowed rules ({e}) — "
              "rule-uncovered subset unavailable")
        return None
    uncovered = set()
    for fid, f in facts.items():
        qa = QAPair(question=f["question"], answer=f["answer"],
                    value=f["answer"])
        if policy._score_one(qa) == policy._default_score:
            uncovered.add(fid)
    return uncovered


def load_results(results_dir, seeds):
    """{(cond, seed): result JSON}; missing cells listed separately."""
    results, missing = {}, []
    for cond in CONDITIONS:
        for seed in seeds:
            path = Path(results_dir) / f"ego_{cond}_s{seed}.json"
            if not path.exists():
                missing.append((cond, seed))
                continue
            try:
                with open(path) as f:
                    res = json.load(f)
                if res.get("quick"):
                    print(f"[Analyze] Skipping quick run {path.name}")
                    missing.append((cond, seed))
                    continue
                results[(cond, seed)] = res
            except (OSError, ValueError) as e:
                print(f"[Analyze] WARNING: unreadable {path.name}: {e}")
                missing.append((cond, seed))
    return results, missing


# ── Per-run metrics (§6) ──

def _snapshot_retained_fids(entries, by_qkey):
    """fact_ids retained in a full-ledger snapshot: (not pruned) OR graduated."""
    fids = set()
    for e in entries:
        if e.get("pruned", False) and not e.get("graduated", False):
            continue
        f = by_qkey.get(_qkey(e["qa"]["question"]))
        if f:
            fids.add(f["fact_id"])
    return fids


def run_metrics(res, ctx):
    """All per-run M1-M10 + gate readouts for one result JSON."""
    facts = ctx["facts"]
    by_qkey = ctx["by_qkey"]
    shift = ctx["shift_session"]
    m = {}

    # M1 — VWR with the simulated achievable-optimum denominator (§6 M1).
    # The harness already computed vwr against the header denominators.
    traj = res.get("vwr_trajectory", [])
    m["m1_final"] = traj[-1]["vwr"] if traj else None
    pre_pts = [t for t in traj if t.get("labels") == "pre"]
    post_pts = [t for t in traj if t.get("labels") == "post"]
    m["m1_pre_shift"] = pre_pts[-1]["vwr"] if pre_pts else None
    m["m1_post_shift"] = post_pts[-1]["vwr"] if post_pts else None
    m["vwr_trajectory"] = [(t.get("after"), t.get("vwr")) for t in traj]

    # M2 — composition, from per_fact_outcomes joined against the stream.
    outcomes = res.get("per_fact_outcomes", {})
    retained = {}
    for fid, f in facts.items():
        o = outcomes.get(fid)
        retained[fid] = 1 if (o and o.get("retained")) else 0
    m["retained_by_fact"] = retained

    cells = {}
    for fid, f in facts.items():
        cells.setdefault(f["cell"], []).append(retained[fid])
    m["cell_survival"] = {c: sum(v) / len(v) for c, v in sorted(cells.items())}
    m["cell_a_survival"] = m["cell_survival"].get("A")
    m["slot_waste"] = sum(1 for fid, r in retained.items()
                          if r and facts[fid]["value_post"] == 0.0)
    uncovered = ctx["rule_uncovered"]
    if uncovered is not None:
        uv = [retained[fid] for fid in uncovered
              if facts[fid]["value_post"] >= VITAL_THRESHOLD]
        m["rule_uncovered_vital_survival"] = (sum(uv) / len(uv)) if uv else None
    else:
        m["rule_uncovered_vital_survival"] = None
    m["admission_rejections"] = len(res.get("admission_rejections", []))

    # M3 — Spearman(assigned priority, value label) over all scored facts:
    # final full-ledger snapshot entries (incl. pruned — they carry the last
    # assigned priority) plus admission-gate rejections (scored, never held).
    pairs = []
    snaps = res.get("ledger_snapshots", [])
    if snaps:
        for e in snaps[-1]["entries"]:
            f = by_qkey.get(_qkey(e["qa"]["question"]))
            prio = e["qa"].get("priority")
            if f and prio is not None:
                pairs.append((float(prio), f["value_post"]))
    for r in res.get("admission_rejections", []):
        f = by_qkey.get(_qkey(r.get("question", "")))
        if f and r.get("priority") is not None:
            label = (f["value_post"] if r.get("session", 99) >= shift
                     else f["value_pre"])
            pairs.append((float(r["priority"]), label))
    m["rho_pairs"] = pairs
    m["m3_rho"] = spearman_rho([p[0] for p in pairs], [p[1] for p in pairs])

    # M4 — cycles-to-revalue (post-shift adaptation, sleeps 2..4).
    flip_up = [fid for fid, f in facts.items()
               if f["cell"] == "G" and f["value_post"] > f["value_pre"]]
    flip_down = [fid for fid, f in facts.items()
                 if f["cell"] == "G" and f["value_post"] < f["value_pre"]]
    m["m4_cycles"] = None
    if flip_up and flip_down:
        for i, tag in enumerate(("sleep_2", "sleep_3", "sleep_4"), start=1):
            snap = next((s for s in snaps if s.get("after") == tag), None)
            if snap is None:
                continue
            held = _snapshot_retained_fids(snap["entries"], by_qkey)
            up_ok = sum(1 for fid in flip_up if fid in held)
            down_ok = sum(1 for fid in flip_down if fid not in held)
            if (up_ok >= M4_REVALUE_FRACTION * len(flip_up) - 1e-9
                    and down_ok >= M4_REVALUE_FRACTION * len(flip_down) - 1e-9):
                m["m4_cycles"] = i
                break

    # M5 — plants & hearsay (synthetic stress test on the ledger arm).
    plant_fids = [fid for fid, f in facts.items()
                  if f["provenance"] == "assistant_generated"]
    hearsay_fids = [fid for fid, f in facts.items()
                    if f["provenance"] == "user_reported_hearsay"]

    def _prios(fids):
        out = []
        for fid in fids:
            o = outcomes.get(fid)
            if o and o.get("final_priority") is not None:
                out.append(float(o["final_priority"]))
        return out

    m["plant_retention"] = (sum(retained[f] for f in plant_fids)
                            / len(plant_fids)) if plant_fids else None
    m["plant_priorities"] = _prios(plant_fids)
    m["hearsay_retention"] = (sum(retained[f] for f in hearsay_fids)
                              / len(hearsay_fids)) if hearsay_fids else None
    m["hearsay_priorities"] = _prios(hearsay_fids)
    m["plants_in_train_total"] = sum(
        (c.get("composition") or {}).get("assistant_generated", 0)
        for c in res.get("training_set_composition", []))
    stale = [fid for fid, f in facts.items()
             if f["cell"] == "F" and f["value_post"] < 0.25]
    corrected = [fid for fid, f in facts.items()
                 if f["cell"] == "F" and f["value_post"] >= 0.25]
    m["stale_evicted"] = (sum(1 - retained[f] for f in stale)
                          / len(stale)) if stale else None
    m["corrected_retained"] = (sum(retained[f] for f in corrected)
                               / len(corrected)) if corrected else None

    # M6 — future-task probe score, commitment probes separately (P6).
    fb = res.get("final_battery", {})
    ft = fb.get("future_task") or {}
    m["m6"] = ft.get("score")
    probes = ft.get("probes", [])
    m["probe_correct"] = {p.get("probe_id"): (1 if p.get("correct") else 0)
                          for p in probes if p.get("probe_id") is not None}
    commit = [p for p in probes if p.get("type") == "commitment"]
    m["m6_commitment"] = (sum(1 for p in commit if p.get("correct"))
                          / len(commit)) if commit else None
    m["commitment_correct"] = {p.get("probe_id"): (1 if p.get("correct") else 0)
                               for p in commit if p.get("probe_id") is not None}

    # M7 — weights-only recall (secondary), value-weighted + split.
    wr = fb.get("weights_recall") or {}
    m["m7"] = wr.get("score")
    wfacts = wr.get("facts", [])
    if wfacts:
        wsum = sum(f.get("value_post", 0.0) for f in wfacts)
        m["m7_value_weighted"] = (sum(f.get("value_post", 0.0)
                                      for f in wfacts if f.get("hit")) / wsum
                                  if wsum else None)
        ret_hits = [f for f in wfacts if retained.get(f.get("fact_id"))]
        ev_hits = [f for f in wfacts if not retained.get(f.get("fact_id"))]
        m["m7_retained"] = (sum(1 for f in ret_hits if f.get("hit"))
                            / len(ret_hits)) if ret_hits else None
        m["m7_evicted"] = (sum(1 for f in ev_hits if f.get("hit"))
                           / len(ev_hits)) if ev_hits else None
    else:
        m["m7_value_weighted"] = m["m7_retained"] = m["m7_evicted"] = None

    # M8 / M9 — contradiction resolution, assertion leakage.
    m["m8"] = (fb.get("contradiction") or {}).get("score")
    m["m9"] = (fb.get("provenance") or {}).get("leakage")

    # M10 — forgetting curve (cell-A recall per cycle, final minus peak).
    curve = []
    for fp in res.get("forgetting_probes", []):
        ps = fp.get("probes", [])
        if ps:
            curve.append(sum(1 for p in ps if p.get("hit")) / len(ps))
    m["m10_curve"] = curve
    m["m10_drop"] = (curve[-1] - max(curve)) if curve else None

    # Gates.
    ppls = [p["ppl"] for p in res.get("ppl_per_cycle", [])
            if p.get("ppl") is not None]
    drifts = [(b - a) / a for a, b in zip(ppls, ppls[1:]) if a]
    m["g1_max_drift"] = max(drifts) if drifts else None
    vlogs = res.get("valuation_logs", [])
    m["g2_fallback_rate"] = None
    if vlogs:
        last = vlogs[-1].get("policy", {})
        scored = last.get("scored_count") or 0
        if scored:
            m["g2_fallback_rate"] = (last.get("fallback_count") or 0) / scored
    m["g4_total_wall"] = (res.get("wall_times") or {}).get("total")
    m["g4_wake_wall"] = (res.get("wall_times") or {}).get("wake_and_sleeps")
    m["assertions_ok"] = all((res.get("harness_assertions") or {}).values())
    return m


# ── Aggregation + contrast machinery ──

# Endpoint registry: item sets drive McNemar/bootstrap; direction 'lt' means
# lower is better (M9 leakage). 'weighted' scales the bootstrap into VWR units.
ENDPOINTS = {
    "m1_final": {"label": "M1 VWR (final)", "items": "all", "weighted": True},
    "m1_post_shift": {"label": "M1 VWR (post-shift)", "items": "all",
                      "weighted": True},
    "cell_a_survival": {"label": "M2 cell-A survival", "items": "cell_a"},
    "rule_uncovered_vital_survival": {"label": "M2 rule-uncovered vital",
                                      "items": "uncovered_vital"},
    "m3_rho": {"label": "M3 Spearman rho", "kind": "rho"},
    "m4_cycles": {"label": "M4 cycles-to-revalue", "direction": "lt"},
    "m6": {"label": "M6 probe score", "items": "probes"},
    "m6_commitment": {"label": "M6 commitment probes",
                      "items": "commitment_probes"},
    "m8": {"label": "M8 contradiction accuracy"},
    "m9": {"label": "M9 assertion leakage", "direction": "lt"},
}


def _item_set(key, ctx):
    facts = ctx["facts"]
    if key == "all":
        return {fid: facts[fid]["value_post"] for fid in facts}
    if key == "cell_a":
        return {fid: 1.0 for fid, f in facts.items() if f["cell"] == "A"}
    if key == "uncovered_vital":
        unc = ctx["rule_uncovered"]
        if unc is None:
            return None
        return {fid: 1.0 for fid in unc
                if facts[fid]["value_post"] >= VITAL_THRESHOLD}
    return None  # probe item sets are read off the runs directly


class World:
    """All computed state the clause evaluator reads (§3 decision rules)."""

    def __init__(self, ctx, metrics, seeds, n_boot):
        self.ctx = ctx
        self.metrics = metrics          # {(cond, seed): run metric dict}
        self.seeds = seeds
        self.n_boot = n_boot
        self.rng = random.Random(RNG_SEED)
        self.voided = set()             # G2-voided conditions
        self.holm_p = {}                # (a, b, endpoint) -> adjusted p
        self._cache = {}
        self.all_contrasts = []         # every evaluated contrast (null reporting)

    def series(self, cond, key):
        out = {}
        for seed in self.seeds:
            m = self.metrics.get((cond, seed))
            if m is not None and m.get(key) is not None:
                out[seed] = m[key]
        return out

    def mean(self, cond, key):
        s = self.series(cond, key)
        return sum(s.values()) / len(s) if s else None

    def contrast(self, a, b, endpoint):
        """Full §9 stats for one pairwise contrast; cached."""
        cache_key = (a, b, endpoint)
        if cache_key in self._cache:
            return self._cache[cache_key]
        spec = ENDPOINTS.get(endpoint, {})
        lower_better = spec.get("direction") == "lt"
        sa, sb = self.series(a, endpoint), self.series(b, endpoint)
        common = sorted(set(sa) & set(sb))
        c = {"a": a, "b": b, "endpoint": endpoint, "n_seeds": len(common),
             "mean_a": self.mean(a, endpoint), "mean_b": self.mean(b, endpoint),
             "delta": None, "p_perm": None, "sign_consistent": None,
             "mcnemar": None, "boot_ci": None, "rho_a": None, "rho_b": None}
        if len(common) >= 2:
            sign = -1.0 if lower_better else 1.0
            diffs = [sign * (sa[s] - sb[s]) for s in common]
            c["delta"] = sum(diffs) / len(diffs)
            c["sign_consistent"] = all(d > 0 for d in diffs)
            if spec.get("kind") == "rho":
                pa, pb = [], []
                for s in common:
                    pa.extend(self.metrics[(a, s)].get("rho_pairs", []))
                    pb.extend(self.metrics[(b, s)].get("rho_pairs", []))
                p, ra, rb = pooled_perm_p(pa, pb, N_PERM_POOLED,
                                          random.Random(RNG_SEED))
                c["p_perm"], c["rho_a"], c["rho_b"] = p, ra, rb
            else:
                pools = []
                for s in common:
                    pool = [sign * self.metrics[(cond, s)][endpoint]
                            for cond in CONDITIONS
                            if (cond, s) in self.metrics
                            and self.metrics[(cond, s)].get(endpoint)
                            is not None]
                    pools.append(pool)
                c["p_perm"] = runlevel_perm_p(diffs, pools,
                                              random.Random(RNG_SEED))
            self._fact_level(c, a, b, endpoint, common, spec)
        self._cache[cache_key] = c
        if c not in self.all_contrasts:
            self.all_contrasts.append(c)
        return c

    def _fact_level(self, c, a, b, endpoint, common, spec):
        """Secondary fact-level stats: exact McNemar on (item, seed) pairs +
        cluster bootstrap on fact_id (§9)."""
        items_key = spec.get("items")
        if not items_key:
            return
        if items_key in ("probes", "commitment_probes"):
            table = ("probe_correct" if items_key == "probes"
                     else "commitment_correct")
            weights, denom = None, None
        else:
            weights = _item_set(items_key, self.ctx)
            if not weights:
                return
            denom = self.ctx["denom_post"] if spec.get("weighted") else None
            table = "retained_by_fact"

        n_b = n_c = 0
        diffs_by_item = {}
        for seed in common:
            ta = self.metrics[(a, seed)].get(table) or {}
            tb = self.metrics[(b, seed)].get(table) or {}
            ids = weights if weights is not None else (set(ta) & set(tb))
            for iid in ids:
                va, vb = ta.get(iid), tb.get(iid)
                if va is None or vb is None:
                    continue
                if va == 1 and vb == 0:
                    n_b += 1
                elif va == 0 and vb == 1:
                    n_c += 1
                w = weights.get(iid, 1.0) if weights is not None else 1.0
                diffs_by_item.setdefault(iid, []).append(w * (va - vb))
        c["mcnemar"] = {"b": n_b, "c": n_c, "p": mcnemar_exact_p(n_b, n_c)}
        per_item = {i: sum(v) / len(v) for i, v in diffs_by_item.items()}
        ci = cluster_bootstrap_ci(per_item, self.n_boot,
                                  random.Random(RNG_SEED), denom=denom)
        if ci:
            c["boot_ci"] = {"est": ci[0], "lo": ci[1], "hi": ci[2]}


# ── §3 decision rules, encoded as data ──
# Clause types the evaluator understands:
#   gt         a > b on endpoint; requires delta > min_delta; optionally
#              Holm-adjusted p < ALPHA ('holm') and all-seed sign ('sign')
#   ge         a >= b on endpoint (delta >= 0; no p requirement)
#   approx     |delta| < margin AND bootstrap CI includes 0 (when available)
#   scalar_le  per-seed metric <= value for cond (None counts as failure)
#   min_of_all cond strictly below every other listed cond, per seed
#   ge_all     cond mean >= every other listed cond's mean on endpoint
#   never_revalues  m4_cycles is None in every available seed for cond
#   g2_below   condition fallback rate below max
#   latency_below   estimated added valuation latency per moment below max
#   no_plants_anywhere  no condition retains a plant or leaks on M9 (P5 SKIP)

NON_ORACLE = ["random", "surprise", "borrowed", "judge", "judge_rules",
              "ego_static"]

PREDICTIONS = [
    {"id": "P0", "kind": "gate",
     "claim": "Validity gate: ORACLE separates from RANDOM and tracks the "
              "achievable optimum",
     "contrast": "ORACLE vs RANDOM", "endpoints": "M6, M1"},
    {"id": "V1", "kind": "check",
     "claim": "Manipulation check: surprise decorrelated from value "
              "(C1 ~= C0 through the pipeline)",
     "contrast": "SURPRISE vs RANDOM", "endpoints": "M1, M2(cell A)",
     "a": "surprise", "b": "random",
     "check_endpoints": ["m1_final", "cell_a_survival"]},
    {"id": "P2", "kind": "generic",
     "claim": "Claims 1+2+5: self-model-referenced valuation beats generic "
              "importance judgment (headline)",
     "contrast": "EGO-STATIC vs JUDGE", "endpoints": "M1, M3, M2(cell A)",
     "supported": {"mode": {"k_of": 2}, "clauses": [
         {"type": "gt", "a": "ego_static", "b": "judge",
          "endpoint": "m1_final", "holm": True, "sign": True},
         {"type": "gt", "a": "ego_static", "b": "judge",
          "endpoint": "m3_rho", "holm": True, "sign": True},
         {"type": "gt", "a": "ego_static", "b": "judge",
          "endpoint": "cell_a_survival", "holm": True, "sign": True}]},
     "falsified": {"mode": "all", "clauses": [
         {"type": "approx", "a": "ego_static", "b": "judge",
          "endpoint": "m1_final"},
         {"type": "approx", "a": "ego_static", "b": "judge",
          "endpoint": "m3_rho"},
         {"type": "approx", "a": "ego_static", "b": "judge",
          "endpoint": "cell_a_survival"}]}},
    {"id": "P2b", "kind": "generic",
     "claim": "Claim 2/7 boundary: self-model-referenced criterion vs any "
              "explicit borrowed criterion via LLM",
     "contrast": "EGO-STATIC vs JUDGE-RULES",
     "endpoints": "M1, M3, M2(rule-uncovered)",
     "supported": {"mode": {"k_of": 1}, "clauses": [
         {"type": "gt", "a": "ego_static", "b": "judge_rules",
          "endpoint": "m1_final", "holm": True, "sign": True},
         {"type": "gt", "a": "ego_static", "b": "judge_rules",
          "endpoint": "rule_uncovered_vital_survival", "holm": True,
          "sign": True}]},
     "falsified": {"mode": "all", "clauses": [
         {"type": "approx", "a": "ego_static", "b": "judge_rules",
          "endpoint": "m1_final"},
         {"type": "approx", "a": "ego_static", "b": "judge_rules",
          "endpoint": "m3_rho"},
         {"type": "approx", "a": "ego_static", "b": "judge_rules",
          "endpoint": "rule_uncovered_vital_survival"}]}},
    {"id": "P3", "kind": "generic",
     "claim": "Claim 7 (static): borrowed ego works but has coverage gaps",
     "contrast": "BORROWED vs SURPRISE; EGO-STATIC vs BORROWED (uncovered)",
     "endpoints": "M1, M2",
     "supported": {"mode": "all", "clauses": [
         {"type": "gt", "a": "borrowed", "b": "surprise",
          "endpoint": "m1_final", "holm": True, "sign": True},
         {"type": "gt", "a": "ego_static", "b": "borrowed",
          "endpoint": "rule_uncovered_vital_survival", "holm": True,
          "sign": True}]},
     "falsified": {"mode": "all", "clauses": [
         {"type": "ge", "a": "borrowed", "b": "ego_static",
          "endpoint": "m1_final"},
         {"type": "ge", "a": "borrowed", "b": "ego_static",
          "endpoint": "rule_uncovered_vital_survival"}]}},
    {"id": "P4", "kind": "generic",
     "claim": "Claim 7 (dynamic): borrowed ego breaks when the frame of "
              "concern shifts",
     "contrast": "EGO-FULL vs BORROWED post-shift",
     "endpoints": "M4, post-shift M1",
     "supported": {"mode": "all", "clauses": [
         {"type": "scalar_le", "cond": "ego_full", "endpoint": "m4_cycles",
          "value": 2, "per_seed": True},
         {"type": "gt", "a": "ego_full", "b": "borrowed",
          "endpoint": "m1_post_shift", "holm": True, "sign": True}]},
     "falsified": {"mode": "any", "clauses": [
         {"type": "approx", "a": "ego_full", "b": "borrowed",
          "endpoint": "m1_post_shift"},
         {"type": "never_revalues", "cond": "ego_full"}]}},
    {"id": "P5", "kind": "generic",
     "claim": "Claim 6 (mechanism demo + behavioral endpoint): provenance "
              "boundary vs model collapse",
     "contrast": "EGO-FULL vs all non-oracle", "endpoints": "M9 (primary), M5",
     "skip": {"mode": "all", "clauses": [{"type": "no_plants_anywhere"}]},
     "supported": {"mode": "all", "clauses": [
         {"type": "min_of_all", "cond": "ego_full", "endpoint": "m9",
          "others": NON_ORACLE, "per_seed": True},
         {"type": "scalar_le", "cond": "ego_full",
          "endpoint": "plants_in_train_total", "value": 0, "per_seed": True},
         {"type": "scalar_le", "cond": "ego_full",
          "endpoint": "plant_priority_max", "value": 0.4, "per_seed": True,
          "allow_missing": True}]},
     "falsified": {"mode": "all", "clauses": [
         {"type": "approx", "a": "ego_full", "b": "surprise",
          "endpoint": "m9"}]}},
    {"id": "P6", "kind": "generic",
     "claim": "Claim 4 (partial): belief revision + provenance + self-indexed "
              "commitments",
     "contrast": "EGO-FULL vs SURPRISE",
     "endpoints": "M8 + stale eviction + commitment probes",
     "supported": {"mode": "all", "clauses": [
         {"type": "gt", "a": "ego_full", "b": "surprise", "endpoint": "m8",
          "holm": True, "sign": True},
         {"type": "ge_all", "cond": "ego_full", "endpoint": "m6_commitment",
          "others": NON_ORACLE}]},
     "falsified": {"mode": "all", "clauses": [
         {"type": "ge", "a": "surprise", "b": "ego_full", "endpoint": "m8"}]}},
    {"id": "P7", "kind": "generic",
     "claim": "Engineering go/no-go: merge the ego module (config-gated)",
     "contrast": "EGO-FULL vs SURPRISE", "endpoints": "M1, M6, G2, G4",
     "supported": {"mode": "all", "clauses": [
         {"type": "ge", "a": "ego_full", "b": "surprise",
          "endpoint": "m1_final"},
         {"type": "ge", "a": "ego_full", "b": "surprise", "endpoint": "m6"},
         {"type": "g2_below", "cond": "ego_full", "max": G2_VOID_RATE},
         {"type": "latency_below", "cond": "ego_full",
          "max": P7_MAX_LATENCY_S}]},
     "falsified": {"mode": "any", "clauses": [
         {"type": "gt", "a": "surprise", "b": "ego_full",
          "endpoint": "m1_final", "min_delta": 0.0},
         {"type": "g2_above", "cond": "ego_full", "min": G2_VOID_RATE}]}},
]

# The 7 primary contrasts (§9 multiplicity): Holm within endpoint family.
PRIMARY_HOLM_CLAUSES = [
    ("P2", "ego_static", "judge", "m1_final"),
    ("P2", "ego_static", "judge", "m3_rho"),
    ("P2", "ego_static", "judge", "cell_a_survival"),
    ("P2b", "ego_static", "judge_rules", "m1_final"),
    ("P2b", "ego_static", "judge_rules", "rule_uncovered_vital_survival"),
    ("P3", "borrowed", "surprise", "m1_final"),
    ("P3", "ego_static", "borrowed", "rule_uncovered_vital_survival"),
    ("P4", "ego_full", "borrowed", "m1_post_shift"),
    ("P5", "ego_full", "surprise", "m9"),
    ("P6", "ego_full", "surprise", "m8"),
    ("P7", "ego_full", "surprise", "m1_final"),
]


def compute_holm(world):
    """Raw permutation ps for the primary clauses, Holm-adjusted within each
    endpoint family (§9)."""
    by_family = {}
    for pred, a, b, ep in PRIMARY_HOLM_CLAUSES:
        c = world.contrast(a, b, ep)
        by_family.setdefault(ep, {})[(a, b, ep)] = c["p_perm"]
    for ep, fam in by_family.items():
        for key, adj in holm_adjust(fam).items():
            world.holm_p[key] = adj


# ── Clause evaluator ──

def eval_clause(cl, world):
    """-> (True | False | None, detail string). None = cannot be decided."""
    t = cl["type"]
    for cond_key in ("a", "b", "cond"):
        cond = cl.get(cond_key)
        if cond and cond in world.voided:
            return None, f"{cond} voided by G2 (fallback > 30%)"

    if t in ("gt", "ge"):
        c = world.contrast(cl["a"], cl["b"], cl["endpoint"])
        if c["delta"] is None:
            return None, f"{cl['a']} vs {cl['b']} on {cl['endpoint']}: missing runs"
        detail = (f"{cl['a']} vs {cl['b']} on {cl['endpoint']}: "
                  f"delta={c['delta']:+.3f}, p_perm={_fmt_p(c['p_perm'])}, "
                  f"sign {c['n_seeds']} seed(s) "
                  f"{'consistent' if c['sign_consistent'] else 'inconsistent'}")
        if t == "ge":
            return c["delta"] >= -1e-9, detail
        ok = c["delta"] > cl.get("min_delta", 0.0)
        if cl.get("sign"):
            ok = ok and bool(c["sign_consistent"])
        if cl.get("holm"):
            adj = world.holm_p.get((cl["a"], cl["b"], cl["endpoint"]),
                                   c["p_perm"])
            detail += f", p_holm={_fmt_p(adj)}"
            if adj is None:
                return None, detail
            ok = ok and adj < ALPHA
        return ok, detail

    if t == "approx":
        c = world.contrast(cl["a"], cl["b"], cl["endpoint"])
        if c["delta"] is None:
            return None, f"{cl['a']} vs {cl['b']} on {cl['endpoint']}: missing runs"
        margin = cl.get("margin", APPROX_MARGIN)
        ok = abs(c["delta"]) < margin
        detail = (f"{cl['a']} ~= {cl['b']} on {cl['endpoint']}: "
                  f"|delta|={abs(c['delta']):.3f} vs {margin}")
        if c.get("boot_ci"):
            ci = c["boot_ci"]
            ok = ok and (ci["lo"] <= 0.0 <= ci["hi"])
            detail += f", CI [{ci['lo']:+.3f}, {ci['hi']:+.3f}]"
        return ok, detail

    if t == "scalar_le":
        vals = world.series(cl["cond"], cl["endpoint"])
        if not vals:
            if cl.get("allow_missing"):
                return None, f"{cl['cond']} {cl['endpoint']}: not measured"
            # m4_cycles None means 'never revalued' — a real failure, but a
            # fully missing series is missing data.
            if not any((cl["cond"], s) in world.metrics for s in world.seeds):
                return None, f"{cl['cond']} {cl['endpoint']}: missing runs"
            return False, f"{cl['cond']} {cl['endpoint']}: never reached"
        ok = all(v <= cl["value"] for v in vals.values())
        if cl.get("per_seed") and cl["endpoint"] == "m4_cycles":
            # seeds present but metric None = never revalued -> fails
            for s in world.seeds:
                if (cl["cond"], s) in world.metrics and s not in vals:
                    ok = False
        return ok, (f"{cl['cond']} {cl['endpoint']}: "
                    f"{sorted(vals.values())} <= {cl['value']}")

    if t == "never_revalues":
        present = [s for s in world.seeds if (cl["cond"], s) in world.metrics]
        if not present:
            return None, f"{cl['cond']} m4_cycles: missing runs"
        vals = world.series(cl["cond"], "m4_cycles")
        ok = all(s not in vals for s in present)
        return ok, f"{cl['cond']} never revalues in {len(present)} seed(s): {ok}"

    if t == "min_of_all":
        ok_all = True
        for seed in world.seeds:
            mv = world.metrics.get((cl["cond"], seed), {}).get(cl["endpoint"])
            if mv is None:
                return None, f"{cl['cond']} {cl['endpoint']} seed {seed}: missing"
            for other in cl["others"]:
                ov = world.metrics.get((other, seed), {}).get(cl["endpoint"])
                if ov is None:
                    return None, f"{other} {cl['endpoint']} seed {seed}: missing"
                if not mv < ov:
                    ok_all = False
        return ok_all, (f"{cl['cond']} strictly lowest {cl['endpoint']} in "
                        f"all seeds: {ok_all}")

    if t == "ge_all":
        mv = world.mean(cl["cond"], cl["endpoint"])
        if mv is None:
            return None, f"{cl['cond']} {cl['endpoint']}: missing"
        for other in cl["others"]:
            ov = world.mean(other, cl["endpoint"])
            if ov is None:
                return None, f"{other} {cl['endpoint']}: missing"
            if mv < ov - 1e-9:
                return False, (f"{cl['cond']} {cl['endpoint']} {mv:.3f} < "
                               f"{other} {ov:.3f}")
        return True, f"{cl['cond']} >= all others on {cl['endpoint']} ({mv:.3f})"

    if t in ("g2_below", "g2_above"):
        rate = world.mean(cl["cond"], "g2_fallback_rate")
        if rate is None:
            return None, f"{cl['cond']} G2 fallback rate: not logged"
        if t == "g2_below":
            return rate < cl["max"], f"G2 fallback {rate:.1%} < {cl['max']:.0%}"
        return rate > cl["min"], f"G2 fallback {rate:.1%} > {cl['min']:.0%}"

    if t == "latency_below":
        # Estimated added valuation latency per moment: mean wake+sleep wall
        # delta vs SURPRISE / 12 moments (valuation latency is not logged
        # per-moment in the result schema; reported as an estimate).
        lat = estimated_latency(world, cl["cond"])
        if lat is None:
            return None, "added valuation latency: not measurable from logs"
        return lat < cl["max"], f"est. added latency {lat:.1f}s/moment < {cl['max']}s"

    if t == "no_plants_anywhere":
        any_signal = False
        seen = False
        for cond in NON_ORACLE + ["ego_full"]:
            for seed in world.seeds:
                m = world.metrics.get((cond, seed))
                if m is None:
                    continue
                seen = True
                if (m.get("plant_retention") or 0) > 0 or (m.get("m9") or 0) > 0:
                    any_signal = True
        if not seen:
            return None, "no runs to check plants"
        return (not any_signal,
                "no condition retains or asserts plants" if not any_signal
                else "plants retained/asserted somewhere")

    return None, f"unknown clause type {t}"


def estimated_latency(world, cond, baseline="surprise", n_moments=12):
    wa = world.mean(cond, "g4_wake_wall")
    wb = world.mean(baseline, "g4_wake_wall")
    if wa is None or wb is None:
        return None
    return max(0.0, (wa - wb) / n_moments)


def eval_block(block, world):
    """Evaluate a clause block -> (True|False|None, [detail lines])."""
    if not block:
        return False, []
    results, details = [], []
    for cl in block["clauses"]:
        ok, detail = eval_clause(cl, world)
        results.append(ok)
        mark = {True: "YES", False: "no", None: "?"}[ok]
        details.append(f"[{mark}] {detail}")
    mode = block.get("mode", "all")
    trues = sum(1 for r in results if r is True)
    falses = sum(1 for r in results if r is False)
    nones = sum(1 for r in results if r is None)
    if mode == "all":
        out = True if trues == len(results) else (None if nones else False)
    elif mode == "any":
        out = True if trues else (None if nones else False)
    else:  # {"k_of": k}
        k = mode["k_of"]
        if trues >= k:
            out = True
        elif trues + nones < k:
            out = False
        else:
            out = None
    return out, details


# ── Gates & verdicts ──

def eval_p0(world):
    """§6 G3 / §3 P0: ORACLE - RANDOM >= 0.15 on M6 AND ORACLE M1 >= 0.95."""
    g3 = world.ctx["g3"]
    m6_o = world.mean("oracle", "m6")
    m6_r = world.mean("random", "m6")
    m1_o = world.mean("oracle", "m1_final")
    details = []
    if m6_o is None or m6_r is None or m1_o is None:
        return "UNKNOWN", ["ORACLE/RANDOM runs missing — gate not evaluable"]
    margin = m6_o - m6_r
    need_m6 = g3.get("oracle_minus_random_m6_min", G3_M6_MARGIN)
    need_m1 = g3.get("oracle_m1_min_ratio", G3_M1_MIN)
    ok_m6 = margin >= need_m6
    ok_m1 = m1_o >= need_m1
    details.append(f"ORACLE - RANDOM on M6 = {margin:+.3f} "
                   f"(need >= +{need_m6}) -> {'OK' if ok_m6 else 'FAIL'}")
    details.append(f"ORACLE M1 = {m1_o:.3f} of achievable optimum "
                   f"(need >= {need_m1}) -> {'OK' if ok_m1 else 'FAIL'}")
    return ("PASS" if (ok_m6 and ok_m1) else "FAIL"), details


def eval_v1(world, pred):
    """§3 V1 manipulation check: C1 ~= C0 -> HELD; C1 > C0 with dVWR > 0.10
    and CI excluding 0 -> FAILED (decorrelation broke in realization)."""
    details = []
    status = "HELD"
    unknown = False
    for ep in pred["check_endpoints"]:
        c = world.contrast(pred["a"], pred["b"], ep)
        if c["delta"] is None:
            details.append(f"[?] {ep}: missing runs")
            unknown = True
            continue
        line = f"{ep}: delta={c['delta']:+.3f}"
        failed_here = False
        if c.get("boot_ci"):
            ci = c["boot_ci"]
            line += f", CI [{ci['lo']:+.3f}, {ci['hi']:+.3f}]"
            failed_here = (ep == "m1_final" and c["delta"] > APPROX_MARGIN
                           and ci["lo"] > 0)
        else:
            failed_here = ep == "m1_final" and c["delta"] > APPROX_MARGIN
        if failed_here:
            status = "FAILED"
        details.append(f"[{'no' if failed_here else 'YES'}] {line}")
    if unknown and status == "HELD":
        return "UNKNOWN", details
    return status, details


def eval_predictions(world):
    """Evaluate every §3 rule; returns [{id, verdict, details, ...}]."""
    compute_holm(world)
    out = []
    p0_status = None
    for pred in PREDICTIONS:
        rec = {"id": pred["id"], "claim": pred["claim"],
               "contrast": pred.get("contrast", ""),
               "endpoints": pred.get("endpoints", ""), "details": []}
        if pred["kind"] == "gate":
            p0_status, details = eval_p0(world)
            rec["verdict"] = {"PASS": "PASS", "FAIL": "FAIL",
                              "UNKNOWN": "INCONCLUSIVE"}[p0_status]
            rec["details"] = details
        elif pred["kind"] == "check":
            status, details = eval_v1(world, pred)
            rec["verdict"] = {"HELD": "HELD", "FAILED": "FAILED",
                              "UNKNOWN": "INCONCLUSIVE"}[status]
            rec["details"] = details
        else:
            skip_ok, skip_details = eval_block(pred.get("skip"), world)
            sup_ok, sup_details = eval_block(pred.get("supported"), world)
            fal_ok, fal_details = eval_block(pred.get("falsified"), world)
            rec["details"] = (["supported-if:"] + ["  " + d for d in sup_details]
                              + ["falsified-if:"] + ["  " + d for d in fal_details])
            if skip_details:
                rec["details"] = (["skip-if:"] + ["  " + d for d in skip_details]
                                  + rec["details"])
            if skip_ok is True:
                rec["verdict"] = "SKIP"
            elif sup_ok is True:
                rec["verdict"] = "SUPPORTED"
            elif fal_ok is True:
                rec["verdict"] = "FALSIFIED"
            else:
                rec["verdict"] = "INCONCLUSIVE"
        out.append(rec)

    # §3 P0: gate failure aborts interpretation of everything downstream.
    if p0_status == "FAIL":
        for rec in out:
            if rec["id"] != "P0":
                rec["verdict"] = "INCONCLUSIVE"
                rec["details"].append(
                    "OVERRIDDEN: P0 sensitivity gate FAILED — no comparison "
                    "is interpretable (§3); fix eval and rerun")
    elif p0_status == "UNKNOWN":
        for rec in out:
            if rec["id"] != "P0":
                rec["details"].append(
                    "CAUTION: P0 gate not evaluable (missing ORACLE/RANDOM "
                    "runs); verdicts provisional")
    return out


def apply_g2_voids(world):
    """§6 G2: > 30% fallback voids the condition; flags for 10-30% and for
    differential rates vs the judge conditions."""
    flags = []
    for cond in CONDITIONS:
        rate = world.mean(cond, "g2_fallback_rate")
        if rate is None:
            continue
        if rate > G2_VOID_RATE:
            world.voided.add(cond)
            flags.append(f"{COND_LABELS[cond]}: fallback {rate:.1%} > 30% — "
                         "CONDITION VOIDED")
        elif rate > G2_SENSITIVITY_RATE:
            flags.append(f"{COND_LABELS[cond]}: fallback {rate:.1%} in 10-30% "
                         "band — run the pre-registered parsed-only "
                         "sensitivity analysis")
    for pair in (("ego_static", "judge"), ("ego_static", "judge_rules")):
        ra = world.mean(pair[0], "g2_fallback_rate")
        rb = world.mean(pair[1], "g2_fallback_rate")
        if ra is not None and rb is not None and abs(ra - rb) > G2_DIFFERENTIAL_PP:
            flags.append(f"|{pair[0]} - {pair[1]}| fallback = "
                         f"{abs(ra - rb):.1%} > 10 pp — confound flag on P2/P2b")
    return flags


# ── Markdown report (notes/133 skeleton, §7.4 step 8) ──

def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _fmt_p(p):
    if p is None:
        return "—"
    return f"{p:.4f}" if p >= 0.0001 else "<0.0001"


def _seed_range(series):
    if not series:
        return "—"
    vals = sorted(series.values())
    mean = sum(vals) / len(vals)
    return f"{mean:.3f} [{vals[0]:.3f}..{vals[-1]:.3f}] (n={len(vals)})"


def _bar(value, lo=-1.0, hi=1.0, width=24):
    if value is None:
        return " " * width
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    n = int(round(frac * width))
    return "#" * n + "." * (width - n)


def _shade(v):
    if v is None:
        return " "
    return " .:-=+*#%@"[min(9, int(v * 9.999))]


def build_report(world, missing, verdicts, g2_flags, results_dir):
    ctx = world.ctx
    lines = []
    add = lines.append
    present = sorted({c for (c, s) in world.metrics})
    n_runs = len(world.metrics)

    add("# EGO-SELECT Results (notes/133) — GENERATED SKELETON")
    add("")
    add(f"Date: {time.strftime('%Y-%m-%d')}")
    add("Model: mlx-community/Llama-3.2-3B-Instruct-4bit")
    add("Hardware: 8GB MacBook Air M3, MLX backend")
    add("Plan: notes/131-ego-selector-experiment (v2, pre-registered)")
    add(f"Runs analyzed: {n_runs}/{len(CONDITIONS) * len(world.seeds)}"
        + (" — PARTIAL MATRIX, affected contrasts marked INCONCLUSIVE"
           if missing else ""))
    add(f"Generated by experiments/analyze_ego.py "
        f"(bootstrap draws={world.n_boot}, rng seed={RNG_SEED}, "
        f"scipy={'yes' if _scipy_stats else 'no — stdlib fallbacks'})")
    add("")

    # Run inventory
    add("## Run inventory")
    add("")
    add("| Condition | " + " | ".join(f"s{s}" for s in world.seeds) + " |")
    add("|---|" + "---|" * len(world.seeds))
    for cond in CONDITIONS:
        row = [("ok" if (cond, s) in world.metrics else "MISSING")
               for s in world.seeds]
        add(f"| {COND_LABELS[cond]} | " + " | ".join(row) + " |")
    add("")

    # Gates
    add("## Gates")
    add("")
    for rec in verdicts:
        if rec["id"] in ("P0", "V1"):
            add(f"**{rec['id']} — {rec['verdict']}** ({rec['claim']})")
            for d in rec["details"]:
                add(f"- {d}")
            add("")
    add("**G1 — PPL drift** (warn > 0.15/cycle; conditions must not differ)")
    add("")
    add("| Condition | max per-cycle drift |")
    add("|---|---|")
    for cond in CONDITIONS:
        add(f"| {COND_LABELS[cond]} | "
            f"{_fmt(world.mean(cond, 'g1_max_drift'))} |")
    add("")
    add("**G2 — valuation parse-fallback rate** (> 30% voids the condition)")
    add("")
    add("| Condition | fallback rate |")
    add("|---|---|")
    for cond in CONDITIONS:
        r = world.mean(cond, "g2_fallback_rate")
        add(f"| {COND_LABELS[cond]} | {_fmt(r)} |")
    for f in g2_flags:
        add(f"- FLAG: {f}")
    if not g2_flags:
        add("- no G2 flags")
    add("")
    add("**G4 — wall clock**")
    add("")
    add("| Condition | total wall (s) | est. added valuation latency/moment |")
    add("|---|---|---|")
    for cond in CONDITIONS:
        lat = estimated_latency(world, cond)
        add(f"| {COND_LABELS[cond]} | {_fmt(world.mean(cond, 'g4_total_wall'), 0)} "
            f"| {_fmt(lat, 1)} s |")
    add("- latency is estimated as (wake+sleep wall vs C1)/12 moments; the "
        "result schema does not log per-moment valuation latency")
    bad = [f"{COND_LABELS[c]} s{s}" for (c, s), m in sorted(world.metrics.items())
           if not m.get("assertions_ok")]
    add(f"- harness assertions: "
        f"{'all OK' if not bad else 'FAILED in ' + ', '.join(bad)}")
    add("")

    # Verdict table
    add("## Verdict table (§3 pre-registered decision rules)")
    add("")
    add("| ID | Contrast | Endpoint(s) | Verdict |")
    add("|---|---|---|---|")
    for rec in verdicts:
        add(f"| {rec['id']} | {rec['contrast']} | {rec['endpoints']} "
            f"| **{rec['verdict']}** |")
    add("")
    for rec in verdicts:
        add(f"### {rec['id']} — {rec['verdict']}")
        add("")
        add(rec["claim"] + ".")
        for d in rec["details"]:
            add(f"- {d}")
        add("")

    # M1
    add("## M1 — Value-weighted retention (VWR)")
    add("")
    add(f"Denominators (simulated achievable optimum, §6): "
        f"pre-shift {_fmt(ctx['denom_pre'])}, post-shift {_fmt(ctx['denom_post'])}")
    add("")
    add("| Condition | pre-shift | post-shift | final |")
    add("|---|---|---|---|")
    for cond in CONDITIONS:
        add(f"| {COND_LABELS[cond]} "
            f"| {_seed_range(world.series(cond, 'm1_pre_shift'))} "
            f"| {_seed_range(world.series(cond, 'm1_post_shift'))} "
            f"| {_seed_range(world.series(cond, 'm1_final'))} |")
    add("")

    # M2 + heatmap
    add("## M2 — Ledger composition")
    add("")
    add("| Condition | cell-A survival | slot waste (value-0 slots) "
        "| rule-uncovered vital | admission rejections |")
    add("|---|---|---|---|---|")
    for cond in CONDITIONS:
        add(f"| {COND_LABELS[cond]} "
            f"| {_seed_range(world.series(cond, 'cell_a_survival'))} "
            f"| {_fmt(world.mean(cond, 'slot_waste'), 1)} "
            f"| {_seed_range(world.series(cond, 'rule_uncovered_vital_survival'))} "
            f"| {_fmt(world.mean(cond, 'admission_rejections'), 1)} |")
    add("")
    add("### For the Paper: per-cell survival heatmap (mean over seeds)")
    add("")
    cell_names = sorted({f["cell"] for f in ctx["facts"].values()})
    if cell_names and present:
        add("| Condition | " + " | ".join(cell_names) + " |")
        add("|---|" + "---|" * len(cell_names))
        for cond in CONDITIONS:
            row = []
            for cell in cell_names:
                vals = [world.metrics[(cond, s)]["cell_survival"].get(cell)
                        for s in world.seeds
                        if (cond, s) in world.metrics
                        and world.metrics[(cond, s)]["cell_survival"].get(cell)
                        is not None]
                v = sum(vals) / len(vals) if vals else None
                row.append(f"{_fmt(v, 2)} `{_shade(v)}`" if v is not None else "—")
            add(f"| {COND_LABELS[cond]} | " + " | ".join(row) + " |")
    else:
        add("(no runs / no stream — heatmap unavailable)")
    add("")

    # M3 + rho bars
    add("## M3 — Selection quality (Spearman rho, priority vs value)")
    add("")
    add("### For the Paper: rho bars")
    add("")
    add("```")
    for cond in CONDITIONS:
        rho = world.mean(cond, "m3_rho")
        add(f"{COND_LABELS[cond]:<18} {_fmt(rho):>7}  "
            f"|{_bar(rho)}|  (-1 .. +1)")
    add("```")
    add("")

    # M4
    add("## M4 — Cycles to revalue (post-shift adaptation)")
    add("")
    add("| Condition | " + " | ".join(f"s{s}" for s in world.seeds) + " |")
    add("|---|" + "---|" * len(world.seeds))
    for cond in CONDITIONS:
        row = []
        for s in world.seeds:
            m = world.metrics.get((cond, s))
            if m is None:
                row.append("—")
            else:
                v = m.get("m4_cycles")
                row.append(str(v) if v is not None else "never")
        add(f"| {COND_LABELS[cond]} | " + " | ".join(row) + " |")
    add("")

    # M5
    add("## M5 — Plant & hearsay handling (synthetic stress test, ledger arm)")
    add("")
    add("| Condition | plant retention | plant priority (mean) "
        "| plants in train.jsonl | hearsay priority (mean) "
        "| stale evicted | corrected retained |")
    add("|---|---|---|---|---|---|---|")
    for cond in CONDITIONS:
        pp, hp = [], []
        for s in world.seeds:
            m = world.metrics.get((cond, s))
            if m:
                pp.extend(m.get("plant_priorities", []))
                hp.extend(m.get("hearsay_priorities", []))
        add(f"| {COND_LABELS[cond]} "
            f"| {_fmt(world.mean(cond, 'plant_retention'))} "
            f"| {_fmt(sum(pp) / len(pp)) if pp else '—'} "
            f"| {_fmt(world.mean(cond, 'plants_in_train_total'), 1)} "
            f"| {_fmt(sum(hp) / len(hp)) if hp else '—'} "
            f"| {_fmt(world.mean(cond, 'stale_evicted'))} "
            f"| {_fmt(world.mean(cond, 'corrected_retained'))} |")
    add("")

    # M6-M10
    add("## M6 — Future-task probe score (behavior level, primary)")
    add("")
    add("| Condition | 22-probe score | commitment probes (2) |")
    add("|---|---|---|")
    for cond in CONDITIONS:
        add(f"| {COND_LABELS[cond]} "
            f"| {_seed_range(world.series(cond, 'm6'))} "
            f"| {_seed_range(world.series(cond, 'm6_commitment'))} |")
    add("")
    add("## M7-M10 — Weights level (secondary, n=3 honesty per §9)")
    add("")
    add("| Condition | M7 recall | M7 value-wtd | M7 retained | M7 evicted "
        "| M8 contradiction | M9 leakage | M10 drop |")
    add("|---|---|---|---|---|---|---|---|")
    for cond in CONDITIONS:
        add(f"| {COND_LABELS[cond]} "
            f"| {_fmt(world.mean(cond, 'm7'))} "
            f"| {_fmt(world.mean(cond, 'm7_value_weighted'))} "
            f"| {_fmt(world.mean(cond, 'm7_retained'))} "
            f"| {_fmt(world.mean(cond, 'm7_evicted'))} "
            f"| {_fmt(world.mean(cond, 'm8'))} "
            f"| {_fmt(world.mean(cond, 'm9'))} "
            f"| {_fmt(world.mean(cond, 'm10_drop'))} |")
    add("")

    # Statistics appendix — every-null reporting
    add("## Statistics appendix (§9) — every evaluated contrast, "
        "nulls included")
    add("")
    add("Primary: run-level permutation permuting the condition-label "
        "assignment within each seed (null diff = two randomly relabeled "
        "runs from that seed's pool; exact when enumerable, seeded MC "
        "otherwise) + seed sign-consistency co-criterion. Secondary: exact "
        "McNemar on (fact_id, seed) pairs and cluster bootstrap on fact_id "
        f"({world.n_boot} draws, seeded). Holm within endpoint family across "
        "the pre-registered primary contrasts.")
    add("")
    add("| Contrast | Endpoint | n seeds | delta | p_perm | p_Holm | sign 3/3 "
        "| McNemar b/c (p) | bootstrap 95% CI |")
    add("|---|---|---|---|---|---|---|---|---|")
    for c in world.all_contrasts:
        holm = world.holm_p.get((c["a"], c["b"], c["endpoint"]))
        mc = c.get("mcnemar")
        ci = c.get("boot_ci")
        mc_s = f"{mc['b']}/{mc['c']} ({_fmt_p(mc['p'])})" if mc else "—"
        ci_s = (f"[{ci['lo']:+.3f}, {ci['hi']:+.3f}]" if ci else "—")
        sign = ("yes" if c["sign_consistent"]
                else ("no" if c["sign_consistent"] is not None else "—"))
        add(f"| {c['a']} vs {c['b']} | {c['endpoint']} | {c['n_seeds']} "
            f"| {_fmt(c['delta'])} | {_fmt_p(c['p_perm'])} | {_fmt_p(holm)} "
            f"| {sign} | {mc_s} | {ci_s} |")
    if not world.all_contrasts:
        add("| (no contrasts evaluable — no runs) | | | | | | | | |")
    add("")

    # Comparison + files
    add("## Comparison vs notes/111 and notes/62")
    add("")
    add("- TODO: fill in after review — LoRA consolidation ceilings from "
        "notes/111 (v7 comprehensive) and per-fact difficulty findings from "
        "notes/62 (H100 results) for context on M7 recall levels.")
    add("- TODO: relate the global-null branch (§9) outcome, if it fired, to "
        "the notes/111 consolidation bottleneck numbers.")
    add("")
    add("## Files")
    add("")
    add(f"- Plan: notes/131-ego-selector-experiment")
    add(f"- Stream: {DEFAULT_STREAM.relative_to(REPO_ROOT)}")
    add(f"- Analyzer: experiments/analyze_ego.py")
    for (cond, seed) in sorted(world.metrics):
        add(f"- experiments/results/ego_{cond}_s{seed}.json")
    for (cond, seed) in missing:
        add(f"- MISSING: experiments/results/ego_{cond}_s{seed}.json")
    add("")
    return "\n".join(lines)


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="EGO-SELECT analysis (notes/131 §3/§6/§9)")
    parser.add_argument("--simulate-optimum", action="store_true",
                        help="Run the §7.1.5 achievable-optimum simulation on "
                             "the frozen stream (updates the stream header)")
    parser.add_argument("--stream", default=str(DEFAULT_STREAM))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--labels-out", default=str(DEFAULT_LABELS))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--seeds", default="41,42,43",
                        help="Comma-separated seed list (default 41,42,43)")
    parser.add_argument("--boot", type=int, default=N_BOOT,
                        help="Cluster-bootstrap draws (default 10000)")
    parser.add_argument("--out", default=None,
                        help="Write the notes/133 skeleton here "
                             "(default: stdout)")
    args = parser.parse_args()

    if args.simulate_optimum:
        cmd_simulate_optimum(args.stream, args.labels_out)
        return

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    ctx = load_context(args.stream, args.corpus)
    results, missing = load_results(args.results_dir, seeds)
    print(f"[Analyze] {len(results)} run(s) loaded, {len(missing)} missing")

    metrics = {}
    for key, res in results.items():
        try:
            m = run_metrics(res, ctx)
            # derived scalar used by the P5 rule
            m["plant_priority_max"] = (max(m["plant_priorities"])
                                       if m["plant_priorities"] else None)
            metrics[key] = m
        except Exception as e:
            print(f"[Analyze] WARNING: metrics failed for {key}: {e} — "
                  "cell treated as missing")
            missing.append(key)

    world = World(ctx, metrics, seeds, args.boot)
    g2_flags = apply_g2_voids(world)
    verdicts = eval_predictions(world)
    report = build_report(world, missing, verdicts, g2_flags, args.results_dir)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(report)
        print(f"[Analyze] Report written to {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
