"""Sleep gates — decide whether a freshly fused model may replace the current one.

Every fuse (full sleep or micro-sleep) produces a *candidate* model. The
candidate is loaded, three gates run, and only if all pass does the
ModelRegistry commit it as the new last-known-good model. Otherwise the
previous model is reloaded and the candidate directory is deleted.

  Gate 1  perplexity   relative PPL increase on reference text vs. the
                       pre-sleep baseline (sleep.maintenance.max_ppl_increase)
  Gate 2  bleed        for each newly trained fact, neighbor questions (same
                       question about a different person) must not start
                       answering with the new fact's value
  Gate 3  regression   a sample of graduated facts must still be recalled
                       (sleep.gates.regression_max_fail_fraction)

Only the bleed gate can name a culprit fact; the runner retries without it.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.memory.facts import QAPair
from src.sleep.paraphrase import ParaphraseGenerator
from src.sleep.recall_match import value_in_response


# ------------------------------------------------------------------ registry

class ModelRegistry:
    """Tracks the last known-good model path and cleans up old fused models.

    `current` is always a model that passed the gates (or the base model).
    Candidates are loaded by the runner; `commit()` promotes one, `rollback()`
    reloads `current`. Only directories under <fused_root>/fused and
    <fused_root>/adapters are ever deleted; the base model never is.
    """

    def __init__(self, backend, base_path: str, fused_root, keep_previous: int = 1):
        self.backend = backend
        self.base_path = str(base_path)
        self.fused_root = Path(fused_root)
        self.keep_previous = max(0, int(keep_previous))
        loaded = getattr(backend, "model_path", None) or getattr(backend, "_model_path", None)
        self.current: str = str(loaded or base_path)
        self.history: List[str] = []   # previous good models, oldest first

    # --- state ---
    @property
    def last_good(self) -> str:
        return self.current

    def reset(self, path: Optional[str] = None):
        self.current = str(path or self.base_path)
        self.history = []

    # --- transitions ---
    def rollback(self) -> str:
        """Reload the last known-good model."""
        self.backend.reload(self.current)
        return self.current

    def commit(self, new_path: str):
        """Promote a candidate that passed the gates; prune older models."""
        new_path = str(new_path)
        if new_path != self.current:
            self.history.append(self.current)
            self.current = new_path
        self.prune()

    def discard(self, *paths):
        """Delete rejected candidate artifacts (fused dir, adapter dir)."""
        for p in paths:
            if p:
                self._remove_if_managed(Path(p))

    def prune(self):
        """Keep current + keep_previous most recent good models; delete the rest."""
        keep = {self._key(self.current)}
        for p in self.history[-self.keep_previous:] if self.keep_previous else []:
            keep.add(self._key(p))
        fused_dir = self.fused_root / "fused"
        if not fused_dir.exists():
            return
        for d in sorted(fused_dir.iterdir()):
            if d.is_dir() and self._key(d) not in keep:
                self._remove_if_managed(d)

    # --- helpers ---
    @staticmethod
    def _key(p) -> str:
        try:
            return str(Path(p).resolve())
        except OSError:
            return str(p)

    def _is_managed(self, p: Path) -> bool:
        managed_roots = [(self.fused_root / "fused"), (self.fused_root / "adapters")]
        rp = self._key(p)
        for root in managed_roots:
            rk = self._key(root)
            if rp != rk and rp.startswith(rk + "/"):
                return True
        return False

    def _remove_if_managed(self, p: Path):
        if self._key(p) in (self._key(self.current), self._key(self.base_path)):
            return
        if not self._is_managed(p) or not p.exists():
            return
        shutil.rmtree(p, ignore_errors=True)
        # Companion adapter / fused directory with the same cycle name
        twin_root = self.fused_root / ("adapters" if p.parent.name == "fused" else "fused")
        twin = twin_root / p.name
        if twin.exists() and self._key(twin) not in (self._key(self.current), self._key(self.base_path)):
            shutil.rmtree(twin, ignore_errors=True)


# --------------------------------------------------------------------- bleed

@dataclass
class BleedProbe:
    """Neighbor questions per fact with their pre-training answers."""
    neighbors: Dict[str, List[dict]] = field(default_factory=dict)  # fact_id -> [{question, before}]
    values: Dict[str, str] = field(default_factory=dict)             # fact_id -> value

    def restrict(self, fact_ids: Sequence[str]) -> "BleedProbe":
        ids = set(fact_ids)
        return BleedProbe(
            neighbors={k: v for k, v in self.neighbors.items() if k in ids},
            values={k: v for k, v in self.values.items() if k in ids},
        )


@dataclass
class GateReport:
    passed: bool
    failed_gate: Optional[str] = None
    culprit_fact_ids: List[str] = field(default_factory=list)
    ppl: dict = field(default_factory=dict)
    bleed: dict = field(default_factory=dict)
    regression: dict = field(default_factory=dict)
    audit_results: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failed_gate": self.failed_gate,
            "culprit_fact_ids": list(self.culprit_fact_ids),
            "ppl": self.ppl,
            "bleed": self.bleed,
            "regression": self.regression,
        }


class SleepGates:
    def __init__(self, config, backend, probe):
        self.backend = backend
        self.probe = probe
        maintenance = config.get("sleep.maintenance", {}) or {}
        self.max_ppl_increase = float(maintenance.get("max_ppl_increase", 0.15))
        g = config.get("sleep.gates", {}) or {}
        self.enabled = bool(g.get("enabled", True))
        self.bleed_neighbors = int(g.get("bleed_neighbors", 2))
        self.regression_max_fail_fraction = float(g.get("regression_max_fail_fraction", 0.5))
        self.system_prompt = config.context.get("system_prompt", "")

    # --- bare-prompt asking (no facts: isolates what the weights say) ---
    def _ask_bare(self, question: str) -> str:
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}]
        prompt = self.backend.apply_chat_template(messages)
        return self.backend.generate(prompt, max_tokens=60, temperature=self.probe.temperature)

    def prepare_bleed(self, entries: Sequence[dict], avoid_questions: Sequence[str] = ()) -> BleedProbe:
        """Generate neighbor questions and record their answers BEFORE training."""
        probe = BleedProbe()
        if not self.enabled or self.bleed_neighbors <= 0:
            return probe
        for entry in entries:
            qa = QAPair.from_dict(entry["qa"])
            if not qa.value:
                continue
            neighbors = ParaphraseGenerator.neighbor_questions(
                qa, self.bleed_neighbors, avoid_questions=avoid_questions)
            recorded = []
            for nq in neighbors:
                before = self._ask_bare(nq)
                # A neighbor that already answers with the value tells us nothing.
                if value_in_response(qa.value, before):
                    continue
                recorded.append({"question": nq, "before": before})
            if recorded:
                probe.neighbors[entry["fact_id"]] = recorded
                probe.values[entry["fact_id"]] = qa.value
        return probe

    # --- the gates ---
    def ppl_gate(self, ppl_baseline, ref_text) -> dict:
        if not ref_text or not ppl_baseline or ppl_baseline <= 0:
            return {"checked": False, "passed": True, "reason": "no reference text / baseline"}
        after = self.backend.compute_perplexity(ref_text)
        increase = (after - ppl_baseline) / ppl_baseline
        passed = increase <= self.max_ppl_increase
        return {"checked": True, "passed": passed, "before": round(ppl_baseline, 3),
                "after": round(after, 3), "increase": round(increase, 4),
                "max_increase": self.max_ppl_increase}

    def bleed_gate(self, bleed: BleedProbe) -> dict:
        if not bleed.neighbors:
            return {"checked": False, "passed": True, "bled": [], "probes": 0}
        bled = []
        details = []
        probes = 0
        for fact_id, items in bleed.neighbors.items():
            value = bleed.values[fact_id]
            for item in items:
                probes += 1
                after = self._ask_bare(item["question"])
                leaked = value_in_response(value, after)
                details.append({"fact_id": fact_id, "question": item["question"],
                                "before": item["before"], "after": after, "leaked": leaked})
                if leaked and fact_id not in bled:
                    bled.append(fact_id)
        return {"checked": True, "passed": not bled, "bled": bled, "probes": probes,
                "details": details}

    def regression_gate(self, audit_entries: Sequence[dict], all_qa: Sequence[QAPair]) -> dict:
        """Audit graduated facts on the candidate model WITHOUT touching the ledger."""
        results = []
        for entry in audit_entries:
            qa = QAPair.from_dict(entry["qa"])
            res = self.probe.test(qa, all_qa)
            results.append({"fact_id": entry["fact_id"], "question": qa.question,
                            "value": qa.value, "passed": res["passed"],
                            "fraction": res["fraction"], "fallback": res["fallback"],
                            "phrasings": res["phrasings"]})
        if not results:
            return {"checked": False, "passed": True, "audited": 0, "failed": 0, "results": []}
        failed = sum(1 for r in results if not r["passed"])
        frac = failed / len(results)
        return {"checked": True, "passed": frac <= self.regression_max_fail_fraction,
                "audited": len(results), "failed": failed, "fail_fraction": round(frac, 3),
                "max_fail_fraction": self.regression_max_fail_fraction, "results": results}

    def run(self, ppl_baseline, ref_text, bleed: BleedProbe,
            audit_entries: Sequence[dict], all_qa: Sequence[QAPair]) -> GateReport:
        """Run all gates on the currently loaded (candidate) model."""
        if not self.enabled:
            reg = self.regression_gate(audit_entries, all_qa)
            return GateReport(passed=True, regression=reg, audit_results=reg.get("results", []))

        ppl = self.ppl_gate(ppl_baseline, ref_text)
        bl = self.bleed_gate(bleed)
        reg = self.regression_gate(audit_entries, all_qa)

        failed_gate = None
        culprits: List[str] = []
        if not ppl["passed"]:
            failed_gate = "perplexity"
        elif not bl["passed"]:
            failed_gate = "bleed"
            culprits = list(bl["bled"])
        elif not reg["passed"]:
            failed_gate = "regression"
        if failed_gate and failed_gate != "bleed" and not bl["passed"]:
            # Another gate failed first, but the bleed culprits are still worth naming
            culprits = list(bl["bled"])

        return GateReport(
            passed=failed_gate is None,
            failed_gate=failed_gate,
            culprit_fact_ids=culprits,
            ppl=ppl, bleed=bl,
            regression={k: v for k, v in reg.items() if k != "results"},
            audit_results=reg.get("results", []),
        )
