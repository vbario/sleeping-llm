"""Nap controller — quick graduation audit of recent facts.

A nap tests recall of the N most recent facts WITHOUT the fact in the
system prompt. This checks whether LoRA has absorbed the facts (graduation
readiness). If a graduated fact fails recall, it's un-graduated.
"""

import re
import time

from src.memory.facts import QAPair


class NapController:
    """Executes nap cycles — quick graduation audit."""

    def __init__(self, config, backend, fact_ledger):
        self.config = config
        self.backend = backend
        self.fact_ledger = fact_ledger

        maintenance = config.get("sleep.maintenance", {}) or {}
        self.audit_count = maintenance.get("nap_audit_count", 5)
        self.system_prompt = config.context.get("system_prompt", "")

    def execute_nap(self, cycle_id) -> dict:
        """Execute a nap — audit N most recent facts for graduation readiness.

        Returns:
            Result dict with audit report
        """
        start_time = time.time()

        active_facts = self.fact_ledger.get_active_facts()
        if not active_facts:
            return {
                "status": "skipped",
                "reason": "No active facts to audit",
                "elapsed_seconds": 0,
            }

        # Sort by most recent, take top N
        sorted_facts = sorted(active_facts, key=lambda e: e["qa"]["timestamp"], reverse=True)
        to_audit = sorted_facts[:self.audit_count]

        all_qa = [QAPair.from_dict(e["qa"]) for e in active_facts]

        passed = 0
        failed = 0
        results = []

        for entry in to_audit:
            qa = QAPair.from_dict(entry["qa"])
            fact_id = entry["fact_id"]

            # Test recall without this fact in system prompt
            recalled = self._test_recall(qa, all_qa)
            self.fact_ledger.update_verification(fact_id, 1.0 if recalled else 0.0)

            if recalled:
                passed += 1
            else:
                failed += 1
                # Un-graduate if it was graduated
                if entry.get("graduated", False):
                    self.fact_ledger.retreat_stage(fact_id)
                    print(f"        Un-graduated: {qa.value}")
                else:
                    self.fact_ledger.record_degrade(fact_id)

            results.append({
                "fact_id": fact_id,
                "question": qa.question,
                "value": qa.value,
                "recalled": recalled,
                "was_graduated": entry.get("graduated", False),
            })

        elapsed = time.time() - start_time
        return {
            "status": "complete",
            "audited": len(to_audit),
            "passed": passed,
            "failed": failed,
            "results": results,
            "elapsed_seconds": round(elapsed, 1),
        }

    def _test_recall(self, qa, all_qa) -> bool:
        """Test if the model can recall a fact without it in the system prompt."""
        other_qa = [f for f in all_qa
                     if f.question.lower().strip() != qa.question.lower().strip()]

        parts = [self.system_prompt]
        if other_qa:
            lines = [f"- {f.answer}" for f in other_qa]
            parts.append("Things you remember about the user:\n" + "\n".join(lines))
        system_content = "\n\n".join(parts)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": qa.question},
        ]
        prompt = self.backend.apply_chat_template(messages)
        response = self.backend.generate(prompt, max_tokens=100, temperature=0.3)

        return self._fuzzy_value_match(qa.value, response)

    def _fuzzy_value_match(self, value: str, response: str) -> bool:
        """Check if key tokens from value appear in the response."""
        response_lower = response.lower()

        if value.lower().strip() in response_lower:
            return True

        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "and", "or", "but", "not", "no", "so", "if", "than", "that",
            "this", "it", "its", "he", "she", "they", "who", "what",
        }
        value_tokens = [w for w in re.findall(r'\w+', value.lower())
                        if w not in stop_words and len(w) > 1]

        if not value_tokens:
            return False

        matched = sum(1 for t in value_tokens if t in response_lower)

        if len(value_tokens) <= 2:
            return matched == len(value_tokens)
        return matched / len(value_tokens) >= 0.6

    def execute_nap_streaming(self, cycle_id):
        """Execute nap with streaming progress. Yields progress dicts."""
        total_steps = 2

        # Step 1: Audit
        yield {"step": 1, "total": total_steps, "label": "Auditing recent facts", "status": "running"}

        active_facts = self.fact_ledger.get_active_facts()
        if not active_facts:
            yield {"step": 1, "total": total_steps, "label": "Auditing recent facts", "status": "done",
                   "detail": "No active facts to audit."}
            return

        sorted_facts = sorted(active_facts, key=lambda e: e["qa"]["timestamp"], reverse=True)
        to_audit = sorted_facts[:self.audit_count]
        all_qa = [QAPair.from_dict(e["qa"]) for e in active_facts]

        passed = 0
        failed = 0
        for entry in to_audit:
            qa = QAPair.from_dict(entry["qa"])
            fact_id = entry["fact_id"]

            recalled = self._test_recall(qa, all_qa)
            self.fact_ledger.update_verification(fact_id, 1.0 if recalled else 0.0)

            if recalled:
                passed += 1
            else:
                failed += 1
                if entry.get("graduated", False):
                    self.fact_ledger.retreat_stage(fact_id)
                else:
                    self.fact_ledger.record_degrade(fact_id)

        yield {"step": 1, "total": total_steps, "label": "Auditing recent facts", "status": "done",
               "detail": f"Audited {len(to_audit)}: {passed} passed, {failed} failed"}

        # Step 2: Report
        yield {"step": 2, "total": total_steps, "label": "Report", "status": "done",
               "detail": f"passed={passed}, failed={failed}"}
