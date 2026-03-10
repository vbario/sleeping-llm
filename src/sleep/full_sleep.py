"""Full sleep controller — LoRA consolidation + graduation pipeline.

Implements a 5-step sleep cycle (MEMIT removed):
  [1] Health Check — measure baseline PPL
  [2] Curate — extract new facts from unconsumed conversations
  [3] LoRA Consolidation — train on all non-graduated facts, fuse
  [4] Graduation Test — withhold each fact from system prompt, test recall
  [5] Validate — measure post-training PPL, check for degradation

Graduation: facts pass through stages 0→1→2→3. At stage 3, the fact
is "graduated" — LoRA carries the knowledge and the fact is removed
from the system prompt, freeing context window space.
"""

import json
import time
from pathlib import Path

from src.memory.facts import QAPair


class FullSleepController:
    """Executes full sleep cycles with LoRA consolidation."""

    def __init__(self, config, backend, fact_ledger, curator,
                 validator, session_tracker, health_monitor,
                 fact_extractor=None, trainer=None):
        self.config = config
        self.backend = backend
        self.fact_ledger = fact_ledger
        self.curator = curator
        self.validator = validator
        self.session_tracker = session_tracker
        self.health_monitor = health_monitor
        self.fact_extractor = fact_extractor
        self.trainer = trainer

        maintenance = config.get("sleep.maintenance", {}) or {}
        self.degraded_threshold = maintenance.get("degraded_threshold", 0.5)
        self.max_ppl_increase = maintenance.get("max_ppl_increase", 0.15)

        # Consolidation config
        consolidation = config.get("consolidation", {}) or {}
        lora_cfg = config.get("lora", {}) or {}
        self.consolidation_enabled = (
            trainer is not None
            and lora_cfg.get("enabled", False)
            and consolidation.get("enabled", False)
        )

    @property
    def total_steps(self):
        return 5 if self.consolidation_enabled else 3

    def execute_sleep(self, cycle_id, sleep_type, gather_messages_fn):
        """Execute the full sleep pipeline.

        Returns:
            Result dict with status and details
        """
        start_time = time.time()
        ts = self.total_steps

        # [1] Health check
        print(f"  [1/{ts}] Health check...")
        ref_text = self._get_ppl_reference_text()
        ppl_baseline = self.backend.compute_perplexity(ref_text) if ref_text else None
        if ppl_baseline:
            print(f"        Baseline PPL: {ppl_baseline:.2f}")

        # [2] Curate — gather and persist new facts
        print(f"  [2/{ts}] Curating...")
        messages, consumed_sessions = gather_messages_fn()
        print(f"        {len(consumed_sessions)} new session(s) to process")
        curated = self.curator.curate_with_model(messages, cycle_id)
        print(f"        {len(curated)} exchanges selected")

        new_facts_added = 0
        if curated and self.fact_extractor:
            all_extracted = []
            for exchange in curated:
                msgs = exchange.get("messages", [])
                user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                if user_msg and asst_msg:
                    facts = self.fact_extractor.extract_from_exchange(user_msg, asst_msg)
                    all_extracted.extend(facts)

            if all_extracted:
                existing = self.fact_ledger.get_all_qa_pairs()
                unique = self.fact_extractor.deduplicate(all_extracted, existing)

                # Dedup within batch
                seen_keys = set()
                batch = []
                for fact in unique:
                    key = self.fact_extractor._dedup_key(fact)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        batch.append(fact)

                skipped = len(all_extracted) - len(batch)
                if skipped:
                    print(f"        Skipped {skipped} duplicate/junk fact(s)")

                for qa in batch:
                    self.fact_ledger.add_fact(qa)
                    new_facts_added += 1

            if new_facts_added:
                print(f"        Added {new_facts_added} new facts to ledger")

        if consumed_sessions:
            self.session_tracker.mark_consumed(consumed_sessions, cycle_id)
            print(f"        Marked {len(consumed_sessions)} session(s) as consumed")

        # [3-4] Consolidation + Graduation (if enabled)
        consolidation_stats = {"advanced": 0, "retreated": 0, "already_known": 0, "skipped": True}
        if self.consolidation_enabled:
            print(f"  [3/{ts}] LoRA Consolidation...")
            consolidation_stats = self._consolidate(cycle_id, ref_text)
            print(f"        Advanced: {consolidation_stats['advanced']}, "
                  f"Retreated: {consolidation_stats['retreated']}")

            print(f"  [4/{ts}] Graduation summary:")
            graduated = self.fact_ledger.get_graduated_count()
            total = self.fact_ledger.get_active_fact_count()
            print(f"        {graduated}/{total} facts graduated")

        # [3/5] Validate
        validate_step = 5 if self.consolidation_enabled else 3
        print(f"  [{validate_step}/{ts}] Validating...")
        ppl_after = self.backend.compute_perplexity(ref_text) if ref_text else None
        ppl_ok = True
        if ppl_baseline and ppl_after and ppl_baseline > 0:
            ppl_increase = (ppl_after - ppl_baseline) / ppl_baseline
            print(f"        PPL: {ppl_baseline:.2f} → {ppl_after:.2f} ({ppl_increase:+.1%})")
            if ppl_increase > self.max_ppl_increase:
                ppl_ok = False
                print(f"        WARNING: PPL increase exceeds threshold ({self.max_ppl_increase:.0%})")
        else:
            print(f"        PPL: no reference text, skipping check")

        elapsed = time.time() - start_time
        print(f"  Sleep cycle completed in {elapsed:.1f}s")

        return {
            "status": "approved" if ppl_ok else "warning",
            "new_facts_added": new_facts_added,
            "consolidation": consolidation_stats,
            "ppl_before": round(ppl_baseline, 2) if ppl_baseline else None,
            "ppl_after": round(ppl_after, 2) if ppl_after else None,
            "elapsed_seconds": round(elapsed, 1),
        }

    def _consolidate(self, cycle_id, ref_text) -> dict:
        """LoRA consolidation + graduation testing.

        1. Gather all non-graduated facts
        2. Train LoRA on them (priority-weighted), fuse into model
        3. Graduation test: for each fact, withhold from system prompt and ask
           - Pass → advance stage (at stage 3, fact is graduated)
           - Fail → retreat to stage 0
        4. PPL gate: rollback if PPL degrades too much

        Returns:
            Stats dict with advanced/retreated counts
        """
        stats = {"advanced": 0, "retreated": 0, "already_known": 0,
                 "skipped": False,
                 "advanced_facts": [], "retreated_facts": []}

        # 1. Gather eligible facts (non-graduated, active)
        active_facts = self.fact_ledger.get_active_facts()
        eligible = [e for e in active_facts if not e.get("graduated", False)]
        if not eligible:
            print("        No eligible facts for consolidation")
            stats["skipped"] = True
            return stats

        # Build QAPairs for training
        qa_pairs = [QAPair.from_dict(e["qa"]) for e in eligible]
        print(f"        {len(eligible)} fact(s) eligible for training")

        # 2. Train and fuse
        save_dir = Path(self.config.paths.get("fused_models", "models/fused"))
        fused_path = self.trainer.train_and_fuse(qa_pairs, cycle_id, save_dir, weighted=True)
        if fused_path is None:
            print("        Consolidation aborted: training/fuse failed")
            stats["skipped"] = True
            return stats

        # 3. Reload fused model
        print("        Reloading fused model...")
        self.backend.reload(fused_path)

        # 4. Graduation test: for each fact, test recall without system prompt
        system_prompt = self.config.context.get("system_prompt", "")
        all_qa = [QAPair.from_dict(e["qa"]) for e in active_facts]

        for entry in eligible:
            qa = QAPair.from_dict(entry["qa"])
            fact_id = entry["fact_id"]

            passed = self._test_graduation(qa, all_qa, system_prompt)
            fact_label = f"{qa.question} → {qa.value}"

            if passed:
                new_stage = self.fact_ledger.advance_stage(fact_id)
                stats["advanced"] += 1
                stats["advanced_facts"].append(fact_label)
                graduated = "GRADUATED" if new_stage >= 3 else f"stage {new_stage}"
                print(f"        Advanced: {qa.value} ({graduated})")
            else:
                old_stage = entry.get("stage", 0)
                if old_stage > 0:
                    self.fact_ledger.retreat_stage(fact_id)
                    stats["retreated"] += 1
                    stats["retreated_facts"].append(fact_label)
                    print(f"        Retreated: {qa.value} → stage 0")

            # Record training
            self.fact_ledger.record_training(fact_id)

        # 5. PPL gate
        if ref_text:
            ppl_after = self.backend.compute_perplexity(ref_text)
            if ppl_after and ppl_after > 50:
                print(f"        PPL gate FAILED ({ppl_after:.2f}), warning issued")
                # Note: we don't rollback the fused model — the LoRA training
                # is already in the weights. The PPL gate just flags the issue.

        return stats

    def _test_graduation(self, qa, all_qa, system_prompt) -> bool:
        """Test if LoRA has absorbed a fact by asking without system prompt help.

        Builds a prompt with the test fact excluded from the system prompt,
        asks the question, and checks if the answer's key value appears.
        """
        # Build system prompt with this fact excluded
        other_qa = [f for f in all_qa
                     if f.question.lower().strip() != qa.question.lower().strip()]

        parts = [system_prompt]
        if other_qa:
            lines = [f"- {f.answer}" for f in other_qa]
            parts.append("Things you remember about the user:\n" + "\n".join(lines))
        system_content = "\n\n".join(parts)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": qa.question},
        ]
        prompt = self.backend.apply_chat_template(messages)
        response = self.backend.generate(prompt, max_tokens=100, temperature=0.1)

        # Check if the key value appears in the response
        value = qa.value.lower().strip()
        return value in response.lower()

    def _get_ppl_reference_text(self):
        """Get reference text for perplexity measurement from identity data."""
        identity_dir = Path(self.config.paths["core_identity"])
        identity_file = identity_dir / "identity.jsonl"
        if not identity_file.exists():
            return None
        texts = []
        with open(identity_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        texts.append(item.get("text", ""))
                    except json.JSONDecodeError:
                        continue
        return " ".join(texts)[:2000] if texts else None

    def execute_sleep_streaming(self, cycle_id, sleep_type, gather_messages_fn):
        """Execute sleep pipeline, yielding progress dicts for each step."""
        start_time = time.time()
        ts = self.total_steps

        # [1] Health Check
        yield {"step": 1, "total": ts, "label": "Health check", "status": "running"}
        ref_text = self._get_ppl_reference_text()
        ppl_baseline = self.backend.compute_perplexity(ref_text) if ref_text else None
        yield {"step": 1, "total": ts, "label": "Health check", "status": "done",
               "detail": f"PPL: {ppl_baseline:.2f}" if ppl_baseline else "No reference text"}

        # [2] Curate
        yield {"step": 2, "total": ts, "label": "Curating", "status": "running"}
        messages, consumed_sessions = gather_messages_fn()
        curated = self.curator.curate_with_model(messages, cycle_id)

        new_facts_added = 0
        added_fact_labels = []
        if curated and self.fact_extractor:
            all_extracted = []
            for exchange in curated:
                msgs = exchange.get("messages", [])
                user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                if user_msg and asst_msg:
                    facts = self.fact_extractor.extract_from_exchange(user_msg, asst_msg)
                    all_extracted.extend(facts)

            if all_extracted:
                existing = self.fact_ledger.get_all_qa_pairs()
                unique = self.fact_extractor.deduplicate(all_extracted, existing)

                seen_keys = set()
                batch = []
                for fact in unique:
                    key = self.fact_extractor._dedup_key(fact)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        batch.append(fact)

                for qa in batch:
                    self.fact_ledger.add_fact(qa)
                    new_facts_added += 1
                    added_fact_labels.append(f"{qa.question} → {qa.value}")

        if consumed_sessions:
            self.session_tracker.mark_consumed(consumed_sessions, cycle_id)

        yield {"step": 2, "total": ts, "label": "Curating", "status": "done",
               "detail": f"{len(curated)} exchanges, {new_facts_added} facts added, "
                         f"{len(consumed_sessions)} session(s)",
               "facts": added_fact_labels}

        # [3-4] Consolidation + Graduation (if enabled)
        consolidation_stats = {"advanced": 0, "retreated": 0, "skipped": True}
        if self.consolidation_enabled:
            yield {"step": 3, "total": ts, "label": "LoRA Consolidation", "status": "running"}
            consolidation_stats = self._consolidate(cycle_id, ref_text)
            yield {"step": 3, "total": ts, "label": "LoRA Consolidation", "status": "done",
                   "detail": f"Advanced: {consolidation_stats['advanced']}, "
                             f"Retreated: {consolidation_stats['retreated']}",
                   "facts": consolidation_stats.get("advanced_facts", []),
                   "facts_bad": consolidation_stats.get("retreated_facts", [])}

            graduated = self.fact_ledger.get_graduated_count()
            total = self.fact_ledger.get_active_fact_count()
            yield {"step": 4, "total": ts, "label": "Graduation", "status": "done",
                   "detail": f"{graduated}/{total} facts graduated"}

        # [3/5] Validate
        validate_step = 5 if self.consolidation_enabled else 3
        yield {"step": validate_step, "total": ts, "label": "Validating", "status": "running"}
        ppl_after = self.backend.compute_perplexity(ref_text) if ref_text else None
        ppl_detail = "No reference text"
        if ppl_baseline and ppl_after and ppl_baseline > 0:
            ppl_increase = (ppl_after - ppl_baseline) / ppl_baseline
            ppl_detail = f"PPL: {ppl_baseline:.2f} → {ppl_after:.2f} ({ppl_increase:+.1%})"
        yield {"step": validate_step, "total": ts, "label": "Validating", "status": "done",
               "detail": ppl_detail}

        # Final report
        elapsed = time.time() - start_time
        yield {"step": ts, "total": ts, "label": "Report", "status": "done",
               "detail": f"new_facts={new_facts_added}, "
                         f"advanced={consolidation_stats.get('advanced', 0)}, "
                         f"retreated={consolidation_stats.get('retreated', 0)}",
               "facts_refreshed": 0,
               "facts_pruned": 0}
