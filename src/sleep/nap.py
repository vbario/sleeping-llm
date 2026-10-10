"""Nap controller — quick graduation audit of recent and graduated facts.

A nap tests recall WITHOUT the fact in the system prompt. This checks
whether LoRA has absorbed the fact (graduation readiness) and, for facts
that already graduated, whether a later LoRA fuse eroded it.

Two samples are audited each nap:
  - the N most recent active (non-graduated) facts   (nap_audit_count)
  - the M stalest graduated facts                     (graduated_audit_count)

A graduated fact that fails retreats one stage: it moves back into
ledger.json, re-enters the system prompt and the next training set, and
its degrade_count is recorded.
"""

import time

from src.memory.facts import QAPair
from src.sleep.probe import RecallProbe


class NapController:
    """Executes nap cycles — quick graduation audit."""

    def __init__(self, config, backend, fact_ledger):
        self.config = config
        self.backend = backend
        self.fact_ledger = fact_ledger

        maintenance = config.get("sleep.maintenance", {}) or {}
        self.audit_count = maintenance.get("nap_audit_count", 5)
        self.graduated_audit_count = maintenance.get("graduated_audit_count", 5)
        self.system_prompt = config.context.get("system_prompt", "")

        grad = config.get("sleep.graduation", {}) or {}
        self.probe = RecallProbe(
            backend,
            system_prompt=self.system_prompt,
            temperature=grad.get("test_temperature", 0.1),
            min_pass_fraction=grad.get("min_pass_fraction", 1.0),
        )

    # ---------------------------------------------------------------- core

    def _select(self):
        """Return (active_sample, graduated_sample, all_active_qa)."""
        active_facts = self.fact_ledger.get_active_facts()
        sorted_facts = sorted(active_facts, key=lambda e: e["qa"]["timestamp"], reverse=True)
        active_sample = sorted_facts[:self.audit_count]
        graduated_sample = self.fact_ledger.get_audit_candidates(self.graduated_audit_count)
        all_qa = [QAPair.from_dict(e["qa"]) for e in active_facts]
        return active_sample, graduated_sample, all_qa

    def _audit(self, entries, all_qa, graduated):
        """Audit a list of entries. Returns (passed, failed, results)."""
        passed = 0
        failed = 0
        results = []
        for entry in entries:
            qa = QAPair.from_dict(entry["qa"])
            fact_id = entry["fact_id"]

            probe = self.probe.test(qa, all_qa)
            recalled = probe["passed"]
            self.fact_ledger.update_verification(fact_id, probe["fraction"])

            new_stage = entry.get("stage", 0)
            if recalled:
                passed += 1
            else:
                failed += 1
                self.fact_ledger.record_degrade(fact_id)
                if graduated:
                    new_stage = self.fact_ledger.retreat_stage(fact_id)
                    print(f"        Un-graduated: {qa.value} → stage {new_stage}")

            results.append({
                "fact_id": fact_id,
                "question": qa.question,
                "value": qa.value,
                "recalled": recalled,
                "fraction": probe["fraction"],
                "fallback": probe["fallback"],
                "phrasings": probe["phrasings"],
                "was_graduated": graduated,
                "stage": new_stage,
            })
        return passed, failed, results

    def execute_nap(self, cycle_id) -> dict:
        """Execute a nap — audit recent active facts and a graduated sample.

        Returns:
            Result dict with audit report
        """
        start_time = time.time()

        active_sample, graduated_sample, all_qa = self._select()
        if not active_sample and not graduated_sample:
            return {
                "status": "skipped",
                "reason": "No facts to audit",
                "elapsed_seconds": 0,
            }

        a_pass, a_fail, a_results = self._audit(active_sample, all_qa, graduated=False)
        g_pass, g_fail, g_results = self._audit(graduated_sample, all_qa, graduated=True)

        elapsed = time.time() - start_time
        return {
            "status": "complete",
            "audited": len(active_sample) + len(graduated_sample),
            "passed": a_pass + g_pass,
            "failed": a_fail + g_fail,
            "graduated_audited": len(graduated_sample),
            "graduated_failed": g_fail,
            "results": a_results + g_results,
            "elapsed_seconds": round(elapsed, 1),
        }

    def _test_recall(self, qa, all_qa) -> bool:
        """Recall without the fact in the prompt, via held-out phrasings (RecallProbe)."""
        return self.probe.test(qa, all_qa)["passed"]

    def execute_nap_streaming(self, cycle_id):
        """Execute nap with streaming progress. Yields progress dicts."""
        total_steps = 2

        # Step 1: Audit
        yield {"step": 1, "total": total_steps, "label": "Auditing facts", "status": "running"}

        active_sample, graduated_sample, all_qa = self._select()
        if not active_sample and not graduated_sample:
            yield {"step": 1, "total": total_steps, "label": "Auditing facts", "status": "done",
                   "detail": "No facts to audit."}
            return

        a_pass, a_fail, _ = self._audit(active_sample, all_qa, graduated=False)
        g_pass, g_fail, g_results = self._audit(graduated_sample, all_qa, graduated=True)

        yield {"step": 1, "total": total_steps, "label": "Auditing facts", "status": "done",
               "detail": (f"Active {len(active_sample)}: {a_pass} passed, {a_fail} failed. "
                          f"Graduated {len(graduated_sample)}: {g_pass} passed, {g_fail} failed"),
               "facts_bad": [f"{r['question']} → {r['value']}" for r in g_results if not r["recalled"]]}

        # Step 2: Report
        yield {"step": 2, "total": total_steps, "label": "Report", "status": "done",
               "detail": (f"passed={a_pass + g_pass}, failed={a_fail + g_fail}, "
                          f"graduated_audited={len(graduated_sample)}, "
                          f"graduated_failed={g_fail}")}
