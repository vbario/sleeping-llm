"""Nap controller — quick audit of recent MEMIT facts.

A nap tests recall of the N most recent facts and flags degraded ones
for the next full sleep. No model changes — measurement only.
"""

import time


class NapController:
    """Executes nap cycles — quick audit of recent MEMIT edits."""

    def __init__(self, config, backend, memit_engine, ledger):
        self.config = config
        self.backend = backend
        self.memit_engine = memit_engine
        self.ledger = ledger

        maintenance = config.get("sleep.maintenance", {}) or {}
        self.audit_count = maintenance.get("nap_audit_count", 5)
        self.degraded_threshold = maintenance.get("degraded_threshold", 0.5)

    def execute_nap(self, cycle_id) -> dict:
        """Execute a nap — audit N most recent facts, no model changes.

        Args:
            cycle_id: Unique identifier for this nap cycle

        Returns:
            Result dict with audit report
        """
        start_time = time.time()

        # Get N most recent active edits
        active_edits = sorted(
            self.memit_engine._active_edits,
            key=lambda e: e.timestamp,
            reverse=True,
        )[:self.audit_count]

        if not active_edits:
            return {
                "status": "skipped",
                "reason": "No active MEMIT edits to audit",
                "elapsed_seconds": 0,
            }

        # Test recall of each fact
        healthy = 0
        degraded = 0
        results = []

        for edit in active_edits:
            recalled = 0
            for fact in edit.facts:
                passed, response = self.memit_engine.test_recall(fact, raw=True)
                if passed:
                    recalled += 1

            recall_rate = recalled / len(edit.facts) if edit.facts else 1.0
            self.ledger.update_verification(edit.edit_id, recall_rate)

            is_healthy = recall_rate >= self.degraded_threshold
            if is_healthy:
                healthy += 1
            else:
                degraded += 1

            results.append({
                "edit_id": edit.edit_id,
                "fact": f"{edit.facts[0].subject} {edit.facts[0].relation}" if edit.facts else "?",
                "recall_rate": round(recall_rate, 2),
                "healthy": is_healthy,
            })

        elapsed = time.time() - start_time
        return {
            "status": "complete",
            "audited": len(active_edits),
            "healthy": healthy,
            "degraded": degraded,
            "results": results,
            "elapsed_seconds": round(elapsed, 1),
        }

    def execute_nap_streaming(self, cycle_id):
        """Execute nap with streaming progress. Yields progress dicts."""
        total_steps = 2

        # Step 1: Audit
        yield {"step": 1, "total": total_steps, "label": "Auditing recent facts", "status": "running"}

        active_edits = sorted(
            self.memit_engine._active_edits,
            key=lambda e: e.timestamp,
            reverse=True,
        )[:self.audit_count]

        if not active_edits:
            yield {"step": 1, "total": total_steps, "label": "Auditing recent facts", "status": "done",
                   "detail": "No active facts to audit."}
            return

        healthy = 0
        degraded = 0
        for edit in active_edits:
            recalled = 0
            for fact in edit.facts:
                passed, _ = self.memit_engine.test_recall(fact, raw=True)
                if passed:
                    recalled += 1
            recall_rate = recalled / len(edit.facts) if edit.facts else 1.0
            self.ledger.update_verification(edit.edit_id, recall_rate)
            if recall_rate >= self.degraded_threshold:
                healthy += 1
            else:
                degraded += 1

        yield {"step": 1, "total": total_steps, "label": "Auditing recent facts", "status": "done",
               "detail": f"Audited {len(active_edits)}: {healthy} healthy, {degraded} degraded"}

        # Step 2: Report
        yield {"step": 2, "total": total_steps, "label": "Report", "status": "done",
               "detail": f"healthy={healthy}, degraded={degraded}"}
