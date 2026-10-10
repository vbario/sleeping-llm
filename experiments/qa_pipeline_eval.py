"""QA pipeline evaluation — what the current prompt-bridge + LoRA lifecycle does.

Natural first-person facts (experiments/data/personal_facts.json) arrive over
simulated sessions. Between sessions a gated sleep runs through the SAME code
the app uses (ConsolidationRunner: replay, paraphrases, train, fuse, gates,
graduation). Two conditions:

  sleep        facts graduate out of the prompt into the weights
  prompt_only  baseline: no training, every fact stays in the prompt

Per cycle (sleep condition):
  graduated_recall  held-out paraphrase recall of currently graduated facts with
                    nothing about them in the prompt (other active facts stay)
  retention         same, only facts graduated in EARLIER cycles
  bleed_rate        neighbor probes (bare prompt) answering with a fact's value
  ppl_drift         relative perplexity change vs the base model
  update_correct    staged corrections: new value recalled AND old value absent
  usage_recall      fact-USING prompts (not recitation) that state the value
  runner stats      status, failed gate, advanced/retreated/ungraduated, attempts
Baseline: prompt_recall / usage_recall with all facts injected.

Sizes:
  small   20 facts, 5 per session, 2 seeds   (3B 4-bit on a laptop)
  large   200 facts, 25 per session, 3 seeds (GPU box)

    python3 experiments/qa_pipeline_eval.py --size small
    python3 experiments/qa_pipeline_eval.py --size large --config experiments/configs/8b_consolidation.yaml
    python3 experiments/qa_pipeline_eval.py --size small --backend stub   # plumbing only, NOT a measurement

Results: experiments/results/qa_pipeline/<tag>/results.json + summary.md
"""

import argparse
import json
import platform
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config  # noqa: E402
from src.memory.facts import FactLedger, QAPair  # noqa: E402
from src.memory.slots import same_slot  # noqa: E402
from src.sleep.consolidation import ConsolidationRunner  # noqa: E402
from src.sleep.gates import ModelRegistry, SleepGates  # noqa: E402
from src.sleep.paraphrase import ParaphraseGenerator  # noqa: E402
from src.sleep.probe import RecallProbe  # noqa: E402
from src.sleep.recall_match import value_in_response  # noqa: E402
from src.sleep.trainer import SleepTrainer  # noqa: E402

SIZES = {
    "small": {"n_facts": 20, "per_session": 5, "seeds": 2, "n_updates": 2, "usage_sample": 5,
              "bleed_neighbors": 2},
    "large": {"n_facts": 200, "per_session": 25, "seeds": 3, "n_updates": 10, "usage_sample": 20,
              "bleed_neighbors": 2},
}

REF_TEXT = (
    "The town library opens at nine and closes at six on weekdays. Its reading room "
    "faces a small courtyard where a plane tree has grown for over a century. Visitors "
    "may borrow up to ten books at a time, and the catalogue can be searched online. "
    "Children's story hour takes place every Saturday morning, and a volunteer-run "
    "repair cafe meets in the basement on the first Sunday of each month."
)


# ----------------------------------------------------------------- helpers

def set_cfg(config, dotted, value):
    keys = dotted.split(".")
    d = config._data
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def pool_to_qa(p):
    return QAPair(question=p["question"], answer=p["answer"], value=p["value"], priority=0.5)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def std(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.pstdev(xs), 4) if len(xs) > 1 else 0.0 if xs else None


# ---------------------------------------------------------- simulated model

class SimulatedBackend:
    """A fake model for exercising the harness WITHOUT a real model.

    It 'learns' the facts in train.jsonl at fuse time (each with p_learn),
    forgets a previously learned fact with p_forget per fuse, bleeds with
    p_bleed, and drifts perplexity +2% per fuse. Fused 'models' are
    directories holding state.json so rollback/reload behave like the real
    thing. Numbers it produces are NOT measurements of anything.
    """

    def __init__(self, seed=0, p_learn=0.7, p_forget=0.1, p_bleed=0.05):
        self.rng = random.Random(seed)
        self.p_learn, self.p_forget, self.p_bleed = p_learn, p_forget, p_bleed
        self._model_path = "simulated-base"
        self.model_lock = None
        self.state = {"known": {}, "fuses": 0}   # question -> answer

    @property
    def model_path(self):
        return self._model_path

    def load(self, path=None):
        self.reload(path)

    def reload(self, path=None):
        self._model_path = path or "simulated-base"
        sp = Path(self._model_path) / "state.json"
        self.state = json.loads(sp.read_text()) if sp.exists() else {"known": {}, "fuses": 0}

    def apply_chat_template(self, messages, for_training=False):
        parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
        if not for_training:
            parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def _facts_in(self, text):
        out = []
        for line in text.splitlines():
            if line.startswith("- "):
                out.append(line[2:].strip())
        return out

    def generate(self, prompt, max_tokens=None, temperature=None, top_p=None):
        system = prompt.split("<|user|>")[0]
        question = prompt.split("<|user|>")[-1].split("<|assistant|>")[0].strip()
        if question.lower().startswith("a person is talking"):   # paraphrase request
            src = question.split("Question:")[-1].strip()
            from src.sleep.paraphrase import to_first_person
            fp = to_first_person(src).rstrip("?")
            return "\n".join([f"Hey, {fp[0].lower() + fp[1:]}?", f"{fp}, do you recall?",
                               "Which one is it for me?"])
        from src.memory.slots import RELATION_WORDS, tokenize
        q_rel, q_content = tokenize(question)
        # prompt bridge: answers listed in the system prompt
        for answer in self._facts_in(system):
            if self._matches(answer, q_rel, q_content):
                return answer
        # weights
        for q, answer in self.state["known"].items():
            if same_slot(q, question, 0.6):
                return answer
        if any(w in question.lower() for w in RELATION_WORDS) and self.rng.random() < self.p_bleed \
                and self.state["known"]:
            return "I think it's " + self.rng.choice(list(self.state["known"].values()))
        return "I'm not sure, you haven't told me that."

    def _matches(self, answer, q_rel, q_content):
        from src.memory.slots import tokenize
        a_rel, a_content = tokenize(answer)
        if a_rel != q_rel:
            return False
        overlap = len(a_content & q_content)
        return overlap >= max(1, min(2, len(q_content) - 1))

    def count_tokens(self, text):
        return len(str(text).split())

    def compute_perplexity(self, text):
        return 10.0 * (1.02 ** self.state["fuses"])

    def train_lora(self, data_path, adapter_path, **kw):
        rows = [json.loads(l) for l in (Path(data_path) / "train.jsonl").read_text().splitlines()]
        learned = {}
        seen = set()
        for r in rows:
            q, a = r["messages"][0]["content"], r["messages"][1]["content"]
            if a in seen:
                continue
            seen.add(a)
            if self.rng.random() < self.p_learn:
                learned[q] = a
        Path(adapter_path).mkdir(parents=True, exist_ok=True)
        (Path(adapter_path) / "learned.json").write_text(json.dumps(learned))

    def fuse_adapter(self, adapter_path, save_path):
        learned = json.loads((Path(adapter_path) / "learned.json").read_text())
        known = {q: a for q, a in self.state["known"].items() if self.rng.random() >= self.p_forget}
        # a new value for a slot replaces the old one
        for q, a in learned.items():
            known = {kq: ka for kq, ka in known.items() if not same_slot(kq, q, 0.6)}
            known[q] = a
        Path(save_path).mkdir(parents=True, exist_ok=True)
        (Path(save_path) / "state.json").write_text(json.dumps(
            {"known": known, "fuses": self.state["fuses"] + 1}))


# --------------------------------------------------------------- the eval

class PipelineEval:
    def __init__(self, config, backend, size, run_dir, pool, seed, condition, verbose=True):
        self.config = config
        self.backend = backend
        self.size = size
        self.seed = seed
        self.condition = condition
        self.verbose = verbose
        self.rng = random.Random(seed)
        self.run_dir = Path(run_dir) / f"{condition}_seed{seed}"
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        self.run_dir.mkdir(parents=True)

        facts = self.rng.sample(pool, size["n_facts"])
        n = size["per_session"]
        self.sessions = [facts[i:i + n] for i in range(0, len(facts), n)]
        self.update_cycle = min(2, len(self.sessions) - 1)
        # corrections target facts from the first session (they had time to graduate)
        upd_candidates = [f for f in self.sessions[0] if f["alternatives"]]
        self.updates = [{"fact": f, "old": f["value"], "new": f["alternatives"][0]}
                        for f in self.rng.sample(upd_candidates, min(size["n_updates"], len(upd_candidates)))]

        set_cfg(config, "paths.fused_models", str(self.run_dir / "fused"))
        set_cfg(config, "sleep.gates.keep_previous_models", 0)
        set_cfg(config, "sleep.gates.bleed_neighbors", size["bleed_neighbors"])
        facts_cfg = config.get("facts", {}) or {}
        self.ledger = FactLedger(str(self.run_dir / "ledger" / "ledger.json"),
                                 slot_threshold=facts_cfg.get("slot_similarity_threshold", 0.6),
                                 correction_priority=facts_cfg.get("correction_priority", 0.9))
        grad = config.get("sleep.graduation", {}) or {}
        self.system_prompt = config.context.get("system_prompt", "")
        self.probe = RecallProbe(backend, self.system_prompt,
                                 temperature=grad.get("test_temperature", 0.1),
                                 min_pass_fraction=grad.get("min_pass_fraction", 1.0))
        self.paraphraser = ParaphraseGenerator(config, backend)
        self.base_path = config.model["path"] if not isinstance(backend, SimulatedBackend) else "simulated-base"
        self.registry = ModelRegistry(backend, self.base_path, self.run_dir / "fused", keep_previous=0)
        self.gates = SleepGates(config, backend, self.probe)
        self.runner = ConsolidationRunner(config, backend, SleepTrainer(config, backend), self.ledger,
                                          self.probe, paraphraser=self.paraphraser,
                                          registry=self.registry, gates=self.gates)
        self.graduated_at = {}       # fact_id -> cycle index when it graduated
        self.by_question = {}        # question -> pool entry

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # --- measurement primitives ---
    def _ask(self, question, context_qa):
        return self.probe.ask(question, context_qa)

    def _recall(self, entries, withhold):
        """Held-out recall for entries. withhold=True removes each fact from the prompt."""
        active_qa = [QAPair.from_dict(e["qa"]) for e in self.ledger.get_active_facts()]
        per_fact = []
        for e in entries:
            qa = QAPair.from_dict(e["qa"])
            if withhold:
                res = self.probe.test(qa, active_qa, negative_values=e.get("negative_values", []))
                per_fact.append({"fact_id": e["fact_id"], "question": qa.question, "value": qa.value,
                                 "passed": res["passed"], "fraction": res["fraction"],
                                 "fallback": res["fallback"], "negative_hit": res["negative_hit"]})
            else:
                ctx = active_qa if any(a.question == qa.question for a in active_qa) else active_qa + [qa]
                phrasings = qa.held_out or [qa.question]
                hits = [value_in_response(qa.value, self._ask(p, ctx)) for p in phrasings]
                frac = sum(hits) / len(hits)
                per_fact.append({"fact_id": e["fact_id"], "question": qa.question, "value": qa.value,
                                 "passed": frac >= self.probe.min_pass_fraction, "fraction": frac,
                                 "fallback": not qa.held_out, "negative_hit": None})
        return {
            "n": len(per_fact),
            "fact_pass_rate": mean([1.0 if r["passed"] else 0.0 for r in per_fact]),
            "phrasing_recall": mean([r["fraction"] for r in per_fact]),
            "per_fact": per_fact,
        }

    def _bleed(self, entries):
        all_q = [qa.question for qa in self.ledger.get_all_qa_pairs()]
        probes, leaks = 0, 0
        for e in entries:
            qa = QAPair.from_dict(e["qa"])
            for nq in ParaphraseGenerator.neighbor_questions(qa, self.size["bleed_neighbors"], all_q):
                probes += 1
                if value_in_response(qa.value, self._ask(nq, [])):
                    leaks += 1
        return {"probes": probes, "leaks": leaks, "rate": round(leaks / probes, 4) if probes else None}

    def _usage(self, entries, withhold):
        active_qa = [QAPair.from_dict(e["qa"]) for e in self.ledger.get_active_facts()]
        cands = [e for e in entries if self.by_question.get(e["qa"]["question"], {}).get("usage")]
        cands = cands[:self.size["usage_sample"]]
        results = []
        for e in cands:
            qa = QAPair.from_dict(e["qa"])
            usage_q = self.by_question[qa.question]["usage"]
            ctx = self.probe.others(qa, active_qa) if withhold else (
                active_qa if any(a.question == qa.question for a in active_qa) else active_qa + [qa])
            resp = self._ask(usage_q, ctx)
            results.append({"question": usage_q, "value": qa.value,
                            "passed": value_in_response(qa.value, resp), "response": resp[:200]})
        return {"n": len(results), "recall": mean([1.0 if r["passed"] else 0.0 for r in results]),
                "per_prompt": results}

    def _updates(self):
        out = []
        for u in self.updates:
            if not u.get("applied"):
                continue
            e = self.ledger.get_entry(u["new_id"])
            if e is None or e.get("superseded"):
                continue
            qa = QAPair.from_dict(e["qa"])
            active_qa = [QAPair.from_dict(x["qa"]) for x in self.ledger.get_active_facts()]
            if self.condition == "sleep":
                # fact withheld: does the model know the NEW value and not the old one?
                res = self.probe.test(qa, active_qa, negative_values=e.get("negative_values", []))
                frac, neg, passed = res["fraction"], res["negative_hit"], res["passed"]
            else:
                # baseline: the corrected fact is in the prompt (old one is superseded, so absent)
                ctx = active_qa if any(a.question == qa.question for a in active_qa) else active_qa + [qa]
                hits, neg = [], None
                for ph in (qa.held_out or [qa.question]):
                    resp = self._ask(ph, ctx)
                    hits.append(value_in_response(qa.value, resp))
                    for old in e.get("negative_values", []):
                        if value_in_response(old, resp):
                            neg = old
                frac = sum(hits) / len(hits)
                passed = frac >= self.probe.min_pass_fraction and neg is None
            out.append({"question": qa.question, "old": u["old"], "new": u["new"],
                        "graduated": e.get("graduated", False), "stage": e.get("stage", 0),
                        "new_recalled": frac, "old_leaked": neg is not None, "correct": passed})
        return {"n": len(out), "correct_rate": mean([1.0 if r["correct"] else 0.0 for r in out]),
                "per_update": out}

    # --- main loop ---
    def run(self):
        self.backend.reload(self.base_path)
        self.registry.reset(self.base_path)
        base_ppl = self.backend.compute_perplexity(REF_TEXT)
        cycles = []
        for ci, session in enumerate(self.sessions):
            t0 = time.time()
            cycle_id = f"c{ci:02d}"
            self.log(f"\n=== [{self.condition} seed={self.seed}] cycle {ci}: +{len(session)} facts ===")
            for p in session:
                self.by_question[p["question"]] = p
                self.ledger.upsert(pool_to_qa(p))
            update_log = []
            if ci == self.update_cycle:
                for u in self.updates:
                    f = u["fact"]
                    new_qa = QAPair(question=f["question"], value=u["new"], priority=0.5,
                                    answer=f["answer"].replace(f["value"], u["new"]))
                    res = self.ledger.upsert(new_qa)
                    u.update({"applied": True, "new_id": res["fact_id"], "status": res["status"]})
                    update_log.append({"question": f["question"], "old": u["old"], "new": u["new"],
                                       "status": res["status"]})
                    self.log(f"  update: {f['question']} {u['old']!r} -> {u['new']!r} ({res['status']})")

            # paraphrases are needed for baseline recall too
            self.paraphraser.ensure(self.ledger, self.ledger.get_active_facts())

            record = {"cycle": ci, "facts_in_ledger": self.ledger.get_active_fact_count()
                      + self.ledger.get_graduated_count(), "updates": update_log}

            if self.condition == "sleep":
                eligible = [e for e in self.ledger.get_active_facts() if not e.get("graduated")]
                ppl_before = self.backend.compute_perplexity(REF_TEXT)
                stats = self.runner.run(cycle_id, eligible, ref_text=REF_TEXT, ppl_baseline=ppl_before)
                for r in stats["graduation_results"]:
                    e = self.ledger.get_entry(r["fact_id"])
                    if e and e.get("graduated") and r["fact_id"] not in self.graduated_at:
                        self.graduated_at[r["fact_id"]] = ci
                for fid in list(self.graduated_at):
                    e = self.ledger.get_entry(fid)
                    if not e or not e.get("graduated"):
                        self.graduated_at.pop(fid, None)
                graduated = self.ledger.get_graduated_facts()
                earlier = [e for e in graduated if self.graduated_at.get(e["fact_id"], ci) < ci]
                trained_ids = {r["fact_id"] for r in stats["graduation_results"]}
                trained = [e for e in self.ledger.get_active_facts() + graduated if e["fact_id"] in trained_ids]
                record.update({
                    "runner": {k: stats.get(k) for k in ("status", "failed_gate", "attempts", "trained",
                                                           "replayed", "paraphrased", "advanced", "retreated",
                                                           "graduated_audited", "graduated_failed",
                                                           "quarantined", "excluded_facts")},
                    "gates": stats.get("gates", []),
                    "graduated_count": len(graduated),
                    "active_count": self.ledger.get_active_fact_count(),
                    "graduated_recall": self._recall(graduated, withhold=True),
                    "retention": self._recall(earlier, withhold=True),
                    "bleed": self._bleed(trained),
                    "usage": self._usage(graduated, withhold=True),
                    "updates_check": self._updates(),
                })
            else:
                live = self.ledger.get_active_facts()
                record.update({
                    "prompt_recall": self._recall(live, withhold=False),
                    "usage": self._usage(live, withhold=False),
                    "updates_check": self._updates(),
                    "bleed": self._bleed(live[-len(session):]),
                })

            ppl = self.backend.compute_perplexity(REF_TEXT)
            record["ppl"] = {"base": round(base_ppl, 3), "now": round(ppl, 3),
                             "drift": round((ppl - base_ppl) / base_ppl, 4) if base_ppl else None}
            record["elapsed_seconds"] = round(time.time() - t0, 1)
            record["model_path"] = str(self.backend.model_path)
            cycles.append(record)
            self.log(self._cycle_line(record))
        return {"condition": self.condition, "seed": self.seed, "cycles": cycles,
                "sessions": [[p["slot_id"] for p in s] for s in self.sessions],
                "updates": [{k: v for k, v in u.items() if k != "fact"} for u in self.updates]}

    def _cycle_line(self, r):
        if self.condition == "sleep":
            return (f"  status={r['runner']['status']} gate={r['runner']['failed_gate']} "
                    f"graduated={r['graduated_count']} active={r['active_count']} "
                    f"grad_recall={r['graduated_recall']['phrasing_recall']} "
                    f"retention={r['retention']['phrasing_recall']} "
                    f"bleed={r['bleed']['rate']} usage={r['usage']['recall']} "
                    f"updates={r['updates_check']['correct_rate']} ppl_drift={r['ppl']['drift']} "
                    f"({r['elapsed_seconds']}s)")
        return (f"  prompt_recall={r['prompt_recall']['phrasing_recall']} usage={r['usage']['recall']} "
                f"updates={r['updates_check']['correct_rate']} ({r['elapsed_seconds']}s)")


# ---------------------------------------------------------------- summary

def summarize(runs):
    out = {}
    for cond in ("sleep", "prompt_only"):
        cond_runs = [r for r in runs if r["condition"] == cond]
        if not cond_runs:
            continue
        n_cycles = max(len(r["cycles"]) for r in cond_runs)
        per_cycle = []
        for ci in range(n_cycles):
            recs = [r["cycles"][ci] for r in cond_runs if ci < len(r["cycles"])]
            def agg(getter):
                vals = []
                for rec in recs:
                    try:
                        vals.append(getter(rec))
                    except (KeyError, TypeError):
                        vals.append(None)
                return {"mean": mean(vals), "std": std(vals), "n": len([v for v in vals if v is not None])}
            row = {"cycle": ci, "seeds": len(recs),
                   "ppl_drift": agg(lambda x: x["ppl"]["drift"]),
                   "usage_recall": agg(lambda x: x["usage"]["recall"]),
                   "update_correct": agg(lambda x: x["updates_check"]["correct_rate"]),
                   "bleed_rate": agg(lambda x: x["bleed"]["rate"])}
            if cond == "sleep":
                row.update({
                    "graduated_count": agg(lambda x: x["graduated_count"]),
                    "graduated_recall": agg(lambda x: x["graduated_recall"]["phrasing_recall"]),
                    "retention": agg(lambda x: x["retention"]["phrasing_recall"]),
                    "rollbacks": sum(1 for x in recs if x["runner"]["status"] == "rolled_back"),
                    "failed_gates": [x["runner"]["failed_gate"] for x in recs if x["runner"]["failed_gate"]],
                })
            else:
                row["prompt_recall"] = agg(lambda x: x["prompt_recall"]["phrasing_recall"])
            per_cycle.append(row)
        out[cond] = per_cycle
    return out


def fmt(m):
    if m is None or m.get("mean") is None:
        return "  —  "
    return f"{m['mean']:.2f}±{m['std']:.2f}"


def write_summary_md(path, meta, summary):
    lines = [f"# QA pipeline eval — {meta['tag']}", "",
             f"- date: {meta['date']}", f"- backend: {meta['backend']}", f"- model: {meta['model']}",
             f"- hardware: {meta['hardware']}", f"- size: {meta['size']} {json.dumps(meta['size_params'])}",
             f"- seeds: {meta['seeds']}", f"- elapsed: {meta['elapsed_seconds']}s", ""]
    if meta["simulated"]:
        lines += ["**WARNING: simulated backend. These numbers exercise the harness only and "
                  "measure nothing about any model.**", ""]
    if "sleep" in summary:
        lines += ["## sleep condition (mean±std over seeds)", "",
                  "| cycle | graduated | grad. recall | retention | bleed | usage | updates | ppl drift | rollbacks |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in summary["sleep"]:
            lines.append(f"| {r['cycle']} | {fmt(r['graduated_count'])} | {fmt(r['graduated_recall'])} | "
                         f"{fmt(r['retention'])} | {fmt(r['bleed_rate'])} | {fmt(r['usage_recall'])} | "
                         f"{fmt(r['update_correct'])} | {fmt(r['ppl_drift'])} | {r['rollbacks']} "
                         f"{r['failed_gates'] or ''} |")
        lines.append("")
    if "prompt_only" in summary:
        lines += ["## prompt_only baseline (all facts in the prompt, no training)", "",
                  "| cycle | prompt recall | usage | updates | bleed |", "|---|---|---|---|---|"]
        for r in summary["prompt_only"]:
            lines.append(f"| {r['cycle']} | {fmt(r['prompt_recall'])} | {fmt(r['usage_recall'])} | "
                         f"{fmt(r['update_correct'])} | {fmt(r['bleed_rate'])} |")
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


# ------------------------------------------------------------------- main

def make_backend(kind, config, seed):
    if kind == "stub":
        return SimulatedBackend(seed=seed)
    if kind == "auto":
        kind = config.model.get("backend", "mlx")
    if kind == "torch":
        from src.backend.torch_backend import TorchBackend
        b = TorchBackend(config)
    else:
        from src.backend.mlx_backend import MLXBackend
        b = MLXBackend(config)
    b.load()
    return b


def hardware_string():
    try:
        import subprocess
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                 text=True).stdout.strip()) // (1024 ** 3)
        return f"{chip}, {mem} GB RAM, {platform.platform()}"
    except Exception:
        return platform.platform()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(project_root / "config.yaml"))
    ap.add_argument("--size", choices=list(SIZES), default="small")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--n-facts", type=int, default=None)
    ap.add_argument("--per-session", type=int, default=None)
    ap.add_argument("--backend", choices=["auto", "mlx", "torch", "stub"], default="auto")
    ap.add_argument("--conditions", default="sleep,prompt_only")
    ap.add_argument("--pool", default=str(project_root / "experiments" / "data" / "personal_facts.json"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    size = dict(SIZES[args.size])
    if args.seeds:
        size["seeds"] = args.seeds
    if args.n_facts:
        size["n_facts"] = args.n_facts
    if args.per_session:
        size["per_session"] = args.per_session

    config = Config(args.config)
    pool = json.loads(Path(args.pool).read_text())
    tag = args.tag or f"{args.size}_{args.backend}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = project_root / "experiments" / "results" / "qa_pipeline" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    runs = []
    backend = None
    for seed in range(size["seeds"]):
        for cond in args.conditions.split(","):
            if backend is None or args.backend == "stub":
                backend = make_backend(args.backend, config, seed)
            ev = PipelineEval(config, backend, size, run_dir, pool, seed, cond, verbose=not args.quiet)
            runs.append(ev.run())
            (run_dir / "results.json").write_text(json.dumps({"runs": runs}, indent=1))

    summary = summarize(runs)
    meta = {
        "tag": tag, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": "stub (SIMULATED — not a measurement)" if args.backend == "stub" else args.backend,
        "simulated": args.backend == "stub",
        "model": "simulated" if args.backend == "stub" else config.model["path"],
        "hardware": hardware_string(), "size": args.size, "size_params": size,
        "seeds": size["seeds"], "conditions": args.conditions.split(","),
        "elapsed_seconds": round(time.time() - t0, 1),
        "config": args.config,
        "lora": config.get("lora", {}), "graduation": config.get("sleep.graduation", {}),
        "gates": config.get("sleep.gates", {}), "maintenance": config.get("sleep.maintenance", {}),
    }
    (run_dir / "results.json").write_text(json.dumps({"meta": meta, "summary": summary, "runs": runs}, indent=1))
    text = write_summary_md(run_dir / "summary.md", meta, summary)
    print("\n" + text)
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
