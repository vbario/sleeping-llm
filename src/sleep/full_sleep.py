"""Full sleep controller — two-phase deep consolidation with trace preservation.

Implements a sleep pipeline with two distinct phases:

  SWS Phase (all sleep types):
    [1] Pre-eval → [2] Curate → [3] Replay → [4] SWS Train → [5] Validate+Consolidate

  REM Phase (deep sleep only):
    [6] REM Generate (multi-fact conversations from consolidated facts)
    [7] REM Train (integration data + identity, lower LR)
    [8] REM Validate (PPL check + recall spot-check)
    [9] Report

Key principles (from forgetting analysis):
  - MEMIT edits never fully vanish — a residual trace remains (palimpsest)
  - Per-fact granularity — some memories consolidate, some don't
  - Snapshot-based rollback — no fragile revert+re-inject
  - SWS writes individual traces; REM integrates them into coherent schemas
"""

import json
import time
from pathlib import Path


class FullSleepController:
    """Executes full sleep cycles with trace-preserving MEMIT consolidation."""

    # Default residual scale for MEMIT edits after LoRA consolidation.
    # At stage 1+, MEMIT retains this fraction of its delta as a "structural echo"
    # while LoRA carries the primary recall signal.
    DEFAULT_RESIDUAL_SCALE = 0.1

    def __init__(self, config, backend, memit_engine, ledger, curator,
                 trainer, replay_buffer, dreamer, validator, checkpoints,
                 session_tracker, health_monitor):
        self.config = config
        self.backend = backend
        self.memit_engine = memit_engine
        self.ledger = ledger
        self.curator = curator
        self.trainer = trainer
        self.replay_buffer = replay_buffer
        self.dreamer = dreamer
        self.validator = validator
        self.checkpoints = checkpoints
        self.session_tracker = session_tracker
        self.health_monitor = health_monitor

        memit_config = config.get("memit", {}) or {}
        self.residual_scale = memit_config.get("residual_scale_after_consolidation",
                                                self.DEFAULT_RESIDUAL_SCALE)

    def execute_sleep(self, cycle_id, sleep_type, gather_messages_fn):
        """Execute the full sleep pipeline with trace-preserving consolidation.

        Args:
            cycle_id: Unique sleep cycle identifier
            sleep_type: "light" or "deep"
            gather_messages_fn: Callable that returns (messages, consumed_sessions)

        Returns:
            Result dict with status and details
        """
        start_time = time.time()
        total_steps = 9 if sleep_type == "deep" else 7

        # [1] Pre-sleep evaluation
        print(f"  [1/{total_steps}] Running pre-sleep evaluation...")
        pre_score = self.validator.evaluate()
        print(f"        Score: {pre_score['score']:.2f} ({pre_score['correct']}/{pre_score['total']})")

        # [2] Curate training data
        print(f"  [2/{total_steps}] Curating training data...")
        messages, consumed_sessions = gather_messages_fn()
        print(f"        {len(consumed_sessions)} new session(s) to process")

        if sleep_type == "deep":
            curated = self.curator.curate_with_model(messages, cycle_id)
        else:
            curated = self.curator.curate_session(messages, cycle_id)
        print(f"        {len(curated)} exchanges selected for training")

        # Pull MEMIT facts from ledger and add to training queue
        memit_facts = self.ledger.get_facts_for_training()
        active_edits = list(self.memit_engine._active_edits) if memit_facts else []
        memit_pairs = []
        if memit_facts:
            memit_pairs = self.curator.triples_to_training_pairs(memit_facts)
            print(f"        + {len(memit_pairs)} MEMIT fact pairs added")

            training_dir = Path(self.config.paths["training"]) / f"cycle_{cycle_id}"
            training_dir.mkdir(parents=True, exist_ok=True)
            train_file = training_dir / "train.jsonl"
            with open(train_file, "a") as f:
                for pair in memit_pairs:
                    text = self.backend.apply_chat_template(pair, for_training=True)
                    f.write(json.dumps({"text": text}) + "\n")

        if not curated and not memit_facts and self.replay_buffer.stats()["count"] == 0:
            print("        No training data and empty replay buffer. Skipping sleep.")
            return {"status": "skipped", "reason": "No data", "facts_consolidated": 0}

        # [3] Replay buffer
        print(f"  [3/{total_steps}] Updating replay buffer...")
        if curated:
            self.replay_buffer.add(curated)
        stats = self.replay_buffer.stats()
        print(f"        Buffer: {stats['count']} items, avg priority: {stats.get('avg_priority', 0):.2f}")

        # Light sleep: skip dreams step (preserves step numbering)
        if sleep_type != "deep":
            print(f"  [4/{total_steps}] Skipping dreams (light sleep)")

        # SWS Training (step 4 for deep, step 5 for light)
        sws_step = 4 if sleep_type == "deep" else 5
        print(f"  [{sws_step}/{total_steps}] SWS Training...")
        weight_snapshot = None
        pre_sleep_scales = {}
        if active_edits:
            weight_snapshot = self.memit_engine.snapshot_target_weights()
            pre_sleep_scales = {e.edit_id: e.scale for e in active_edits}
            print(f"        Snapshot: {len(weight_snapshot)} target layers saved")

        adapter_path = self.trainer.train(cycle_id, sleep_type)
        if adapter_path is None:
            print("        No training data available. Skipping.")
            return {"status": "skipped", "reason": "No training data", "facts_consolidated": 0}
        print(f"        Adapter saved: {adapter_path}")

        # Consolidation (step 5 for deep, step 6 for light)
        consolidate_step = 5 if sleep_type == "deep" else 6
        print(f"  [{consolidate_step}/{total_steps}] Validating...")
        facts_consolidated = self._validate_and_consolidate(
            active_edits, pre_sleep_scales, weight_snapshot,
            pre_score, consumed_sessions, curated, memit_facts,
            cycle_id, sleep_type,
        )

        # REM phase (deep sleep only, steps 6-8)
        rem_result = None
        if sleep_type == "deep" and self.config.get("rem.enabled", True):
            rem_result = self._execute_rem_phase(
                cycle_id, active_edits, curated, memit_facts, total_steps,
            )

        # Report
        elapsed = time.time() - start_time
        print(f"  [{total_steps}/{total_steps}] Sleep cycle completed in {elapsed:.1f}s")

        result = {
            "status": "approved" if facts_consolidated >= 0 else "rejected",
            "pre_score": pre_score["score"],
            "post_score": pre_score["score"],
            "curated_count": len(curated),
            "memit_facts": len(memit_facts),
            "facts_consolidated": max(0, facts_consolidated),
            "elapsed_seconds": round(elapsed, 1),
        }
        if rem_result:
            result["rem"] = rem_result
        return result

    def _execute_rem_phase(self, cycle_id, active_edits, curated, memit_facts, total_steps):
        """Execute REM integration phase — deep sleep only.

        Generates multi-fact conversations from consolidated facts, trains with
        lower LR, and validates via PPL + recall spot-check. Rolls back to
        post-SWS state if REM hurts the model.

        Returns:
            dict with REM phase results
        """
        # Measure post-SWS PPL as baseline for comparison
        ref_text = self._get_ppl_reference_text()
        sws_ppl = self.backend.compute_perplexity(ref_text) if ref_text else None

        # Snapshot post-SWS MEMIT weights for rollback
        sws_snapshot = self.memit_engine.snapshot_target_weights() if active_edits else None

        # [6] REM Generate
        print(f"  [6/{total_steps}] REM: Generating integration data...")

        # Collect stage 1+ facts (consolidated or consolidating) + SWS facts
        consolidated_facts = []
        for edit in self.memit_engine._active_edits:
            if edit.consolidation_stage >= 1:
                consolidated_facts.extend(edit.facts)
        if memit_facts:
            existing = {(f.subject, f.relation, f.object) for f in consolidated_facts}
            for fact in memit_facts:
                if (fact.subject, fact.relation, fact.object) not in existing:
                    consolidated_facts.append(fact)

        if not consolidated_facts:
            print("        No consolidated facts for REM integration. Skipping.")
            return {"status": "skipped", "reason": "No consolidated facts"}

        recent = [ex["messages"] for ex in curated[:10]] if curated else []
        integrations = self.dreamer.dream_integration(consolidated_facts, recent)
        print(f"        Generated {len(integrations)} integration conversations")

        if not integrations:
            print("        No integration data generated. Skipping REM.")
            return {"status": "skipped", "reason": "No integration data"}

        # Write REM training data
        rem_dir = Path(self.config.paths["training"]) / f"rem_{cycle_id}"
        rem_dir.mkdir(parents=True, exist_ok=True)
        rem_data = self.dreamer.dream_to_training_data(integrations, self.backend)
        with open(rem_dir / "train.jsonl", "w") as f:
            for item in rem_data:
                f.write(json.dumps(item) + "\n")

        # [7] REM Train
        print(f"  [7/{total_steps}] REM: Training on integration data...")
        adapter_path = self.trainer.train_rem(cycle_id, rem_dir)
        if adapter_path is None:
            print("        No REM training data. Skipping.")
            return {"status": "skipped", "reason": "No REM training data"}
        print(f"        REM adapter: {adapter_path}")

        # [8] REM Validate
        print(f"  [8/{total_steps}] REM: Validating...")
        max_ppl_increase = self.config.get("rem.max_ppl_increase", 0.10)

        # PPL check
        rem_ppl = self.backend.compute_perplexity(ref_text) if ref_text else None
        ppl_ok = True
        if sws_ppl and rem_ppl and sws_ppl > 0:
            ppl_increase = (rem_ppl - sws_ppl) / sws_ppl
            print(f"        PPL: {sws_ppl:.2f} → {rem_ppl:.2f} ({ppl_increase:+.1%})")
            if ppl_increase > max_ppl_increase:
                ppl_ok = False
                print(f"        PPL increase exceeds threshold ({max_ppl_increase:.0%})")
        else:
            print("        PPL: no reference text available, skipping check")

        # Recall spot-check on consolidated facts
        recall_ok = True
        sample = consolidated_facts[:5]
        recalled = 0
        for fact in sample:
            passed, _ = self.memit_engine.test_recall(fact)
            status = "OK" if passed else "MISS"
            print(f"        {status}: {fact.subject} {fact.relation} → {fact.object}")
            if passed:
                recalled += 1
        recall_rate = recalled / len(sample) if sample else 1.0
        print(f"        Recall: {recalled}/{len(sample)} ({recall_rate:.0%})")
        if recall_rate < 0.5:
            recall_ok = False

        if ppl_ok and recall_ok:
            print("        REM APPROVED")
            return {
                "status": "approved",
                "integrations": len(integrations),
                "sws_ppl": round(sws_ppl, 2) if sws_ppl else None,
                "rem_ppl": round(rem_ppl, 2) if rem_ppl else None,
                "recall_rate": round(recall_rate, 2),
            }
        else:
            reason = "PPL increase" if not ppl_ok else "Recall dropped"
            print(f"        REM REJECTED ({reason}) — rolling back to post-SWS state")

            # Rollback: reload model from latest checkpoint (post-SWS)
            latest = self.checkpoints.get_latest()
            if latest:
                self.backend.reload(latest["path"])
            if self.memit_engine.enabled and hasattr(self.backend, "dequantize_layer"):
                self.memit_engine._dequantize_target_layers()
            if sws_snapshot:
                self.memit_engine.restore_target_weights(sws_snapshot)

            return {
                "status": "rejected",
                "reason": reason,
                "integrations": len(integrations),
                "sws_ppl": round(sws_ppl, 2) if sws_ppl else None,
                "rem_ppl": round(rem_ppl, 2) if rem_ppl else None,
                "recall_rate": round(recall_rate, 2),
            }

    def _get_ppl_reference_text(self):
        """Get reference text for perplexity measurement.

        Uses identity data as a stable reference that shouldn't degrade.
        """
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

    def _validate_and_consolidate(self, active_edits, pre_sleep_scales,
                                   weight_snapshot, pre_score,
                                   consumed_sessions, curated, memit_facts,
                                   cycle_id, sleep_type):
        """Run benchmark validation and per-fact consolidation.

        Returns:
            Number of facts consolidated (>= 0 on approval, -1 on rejection)
        """
        # Scale all active MEMIT edits to 0.0 to isolate pure LoRA for testing
        if active_edits:
            for edit in active_edits:
                self.memit_engine.scale_edit(edit, 0.0)
            print(f"        Scaled {len(active_edits)} MEMIT edits to 0.0 for pure LoRA test")

        # Benchmark validation
        post_score = self.validator.evaluate()
        print(f"        Post-sleep score: {post_score['score']:.2f} ({post_score['correct']}/{post_score['total']})")
        validation = self.validator.validate_sleep(pre_score, post_score)

        if not validation["approved"]:
            print(f"        REJECTED: {validation['reason']}")
            return self._handle_rejection(active_edits, pre_sleep_scales,
                                          weight_snapshot)

        print(f"        APPROVED: {validation['reason']}")

        # Save checkpoint
        self.checkpoints.save_checkpoint(cycle_id, metadata={
            "sleep_type": sleep_type,
            "pre_score": pre_score["score"],
            "post_score": post_score["score"],
            "curated_count": len(curated),
            "memit_facts": len(memit_facts),
        })

        if consumed_sessions:
            self.session_tracker.mark_consumed(consumed_sessions, cycle_id)
            print(f"        Marked {len(consumed_sessions)} session(s) as consumed")

        # Per-fact consolidation
        facts_consolidated = 0
        if active_edits:
            facts_consolidated = self._per_fact_consolidation(active_edits, pre_sleep_scales)

        return facts_consolidated

    def _per_fact_consolidation(self, active_edits, pre_sleep_scales):
        """Test recall of each fact with pure LoRA (MEMIT at 0.0) and advance stages.

        Returns count of facts that advanced stage.
        """
        consolidated = 0
        for edit in active_edits:
            # Test each fact in this edit
            all_recalled = True
            for fact in edit.facts:
                passed, response = self.memit_engine.test_recall(fact)
                status = "OK" if passed else "MISS"
                print(f"        {status}: {fact.subject} {fact.relation} → {fact.object}")
                if not passed:
                    all_recalled = False

            if all_recalled:
                # LoRA carries this fact — advance consolidation stage
                old_stage = edit.consolidation_stage
                if old_stage == 0:
                    # Stage 0 → 1: scale to residual, mark consolidating
                    edit.consolidation_stage = 1
                    self.memit_engine.scale_edit(edit, self.residual_scale)
                    self.ledger.update_scale(edit.edit_id, self.residual_scale, 1)
                    print(f"          → stage 0→1 (scale {self.residual_scale})")
                    consolidated += 1
                elif old_stage == 1:
                    # Stage 1 → 2: consolidated. Keep residual trace.
                    edit.consolidation_stage = 2
                    # Scale stays at residual_scale (already there from previous cycle)
                    self.memit_engine.scale_edit(edit, self.residual_scale)
                    self.ledger.update_scale(edit.edit_id, self.residual_scale, 2)
                    print(f"          → stage 1→2 (consolidated, residual trace kept)")
                    consolidated += 1
            else:
                # Fact not recalled by LoRA alone — restore to pre-sleep scale
                pre_scale = pre_sleep_scales.get(edit.edit_id, 1.0)
                self.memit_engine.scale_edit(edit, pre_scale)
                print(f"          → kept at scale {pre_scale} (stage {edit.consolidation_stage})")

        print(f"        Consolidated {consolidated}/{len(active_edits)} edits this cycle")
        return consolidated

    def _handle_rejection(self, active_edits, pre_sleep_scales, weight_snapshot):
        """Handle benchmark rejection: restore exact pre-sleep state.

        Returns -1 to signal rejection.
        """
        print("        Rolling back to original model...")

        # Reload pre-training model to undo the LoRA merge
        latest = self.checkpoints.get_latest()
        if latest:
            self.backend.reload(latest["path"])
        else:
            self.backend.reload(self.config.model["path"])

        # Dequantize MEMIT target layers (needed after model reload)
        if self.memit_engine.enabled and hasattr(self.backend, "dequantize_layer"):
            self.memit_engine._dequantize_target_layers()

        # Restore MEMIT weights from snapshot (exact pre-sleep state)
        if weight_snapshot:
            self.memit_engine.restore_target_weights(weight_snapshot)
            # Restore in-memory edit scales
            for edit in active_edits:
                pre_scale = pre_sleep_scales.get(edit.edit_id, 1.0)
                edit.scale = pre_scale
                self.ledger.update_scale(edit.edit_id, pre_scale, edit.consolidation_stage)
            print(f"        Restored {len(active_edits)} MEMIT edits from snapshot")

        print("        Rollback complete.")
        return -1

    def execute_sleep_streaming(self, cycle_id, sleep_type, gather_messages_fn):
        """Execute sleep pipeline, yielding progress dicts for each step."""
        start_time = time.time()
        total_steps = 9 if sleep_type == "deep" else 7

        # [1] Pre-sleep evaluation
        yield {"step": 1, "total": total_steps, "label": "Pre-sleep evaluation", "status": "running"}
        pre_score = self.validator.evaluate()
        yield {"step": 1, "total": total_steps, "label": "Pre-sleep evaluation", "status": "done",
               "detail": f"Score: {pre_score['score']:.2f} ({pre_score['correct']}/{pre_score['total']})"}

        # [2] Curate training data
        yield {"step": 2, "total": total_steps, "label": "Curating training data", "status": "running"}
        messages, consumed_sessions = gather_messages_fn()

        if sleep_type == "deep":
            curated = self.curator.curate_with_model(messages, cycle_id)
        else:
            curated = self.curator.curate_session(messages, cycle_id)

        # Pull MEMIT facts
        memit_facts = self.ledger.get_facts_for_training()
        active_edits = list(self.memit_engine._active_edits) if memit_facts else []
        memit_pairs = []
        if memit_facts:
            memit_pairs = self.curator.triples_to_training_pairs(memit_facts)
            training_dir = Path(self.config.paths["training"]) / f"cycle_{cycle_id}"
            training_dir.mkdir(parents=True, exist_ok=True)
            train_file = training_dir / "train.jsonl"
            with open(train_file, "a") as f:
                for pair in memit_pairs:
                    text = self.backend.apply_chat_template(pair, for_training=True)
                    f.write(json.dumps({"text": text}) + "\n")

        detail = f"{len(curated)} exchanges, {len(consumed_sessions)} session(s)"
        if memit_facts:
            detail += f", +{len(memit_facts)} MEMIT facts"
        yield {"step": 2, "total": total_steps, "label": "Curating training data", "status": "done",
               "detail": detail}

        if not curated and not memit_facts and self.replay_buffer.stats()["count"] == 0:
            yield {"step": 2, "total": total_steps, "label": "Curating training data", "status": "done",
                   "detail": "No new data and empty replay buffer. Skipping."}
            return

        if not curated and not memit_facts:
            yield {"step": 2, "total": total_steps, "label": "Curating training data", "status": "done",
                   "detail": "No new data. Consolidating from replay buffer."}

        # [3] Replay buffer
        yield {"step": 3, "total": total_steps, "label": "Updating replay buffer", "status": "running"}
        if curated:
            self.replay_buffer.add(curated)
        stats = self.replay_buffer.stats()
        yield {"step": 3, "total": total_steps, "label": "Updating replay buffer", "status": "done",
               "detail": f"{stats['count']} items, avg priority: {stats.get('avg_priority', 0):.2f}"}

        # Light sleep: skip dreams placeholder
        if sleep_type != "deep":
            yield {"step": 4, "total": total_steps, "label": "Dreams", "status": "done",
                   "detail": "Skipped (light sleep)"}

        # SWS Training (step 4 for deep, step 5 for light)
        sws_step = 4 if sleep_type == "deep" else 5
        yield {"step": sws_step, "total": total_steps, "label": "SWS Training", "status": "running"}

        weight_snapshot = None
        pre_sleep_scales = {}
        if active_edits:
            weight_snapshot = self.memit_engine.snapshot_target_weights()
            pre_sleep_scales = {e.edit_id: e.scale for e in active_edits}

        adapter_path = self.trainer.train(cycle_id, sleep_type)
        if adapter_path is None:
            yield {"step": sws_step, "total": total_steps, "label": "SWS Training", "status": "done",
                   "detail": "No data available. Skipped."}
            return
        yield {"step": sws_step, "total": total_steps, "label": "SWS Training", "status": "done",
               "detail": "Adapter saved"}

        # Consolidation (step 5 for deep, step 6 for light)
        consolidate_step = 5 if sleep_type == "deep" else 6
        yield {"step": consolidate_step, "total": total_steps, "label": "Consolidating", "status": "running"}

        # Scale MEMIT to 0.0 to isolate pure LoRA
        if active_edits:
            for edit in active_edits:
                self.memit_engine.scale_edit(edit, 0.0)

        post_score = self.validator.evaluate()
        validation = self.validator.validate_sleep(pre_score, post_score)

        if validation["approved"]:
            self.checkpoints.save_checkpoint(cycle_id, metadata={
                "sleep_type": sleep_type,
                "pre_score": pre_score["score"],
                "post_score": post_score["score"],
                "curated_count": len(curated),
                "memit_facts": len(memit_facts),
            })

            if consumed_sessions:
                self.session_tracker.mark_consumed(consumed_sessions, cycle_id)

            # Per-fact consolidation
            facts_consolidated = 0
            if active_edits:
                facts_consolidated = self._per_fact_consolidation(active_edits, pre_sleep_scales)

            consolidate_detail = (f"APPROVED ({post_score['score']:.2f}). "
                                  f"{facts_consolidated} facts consolidated. {validation['reason']}")
        else:
            facts_consolidated = 0
            self._handle_rejection(active_edits, pre_sleep_scales, weight_snapshot)
            consolidate_detail = f"REJECTED ({post_score['score']:.2f}). {validation['reason']}. Rolled back."

        yield {"step": consolidate_step, "total": total_steps, "label": "Consolidating", "status": "done",
               "detail": consolidate_detail}

        # REM phase (deep sleep only, steps 6-8)
        rem_result = None
        if sleep_type == "deep" and self.config.get("rem.enabled", True):
            # [6] REM Generate
            yield {"step": 6, "total": total_steps, "label": "REM: Generating", "status": "running"}

            consolidated_facts = []
            for edit in self.memit_engine._active_edits:
                if edit.consolidation_stage >= 1:
                    consolidated_facts.extend(edit.facts)
            if memit_facts:
                existing = {(f.subject, f.relation, f.object) for f in consolidated_facts}
                for fact in memit_facts:
                    if (fact.subject, fact.relation, fact.object) not in existing:
                        consolidated_facts.append(fact)

            if not consolidated_facts:
                yield {"step": 6, "total": total_steps, "label": "REM: Generating", "status": "done",
                       "detail": "No consolidated facts. Skipped."}
                rem_result = {"status": "skipped"}
            else:
                ref_text = self._get_ppl_reference_text()
                sws_ppl = self.backend.compute_perplexity(ref_text) if ref_text else None
                sws_snapshot = self.memit_engine.snapshot_target_weights() if active_edits else None

                recent = [ex["messages"] for ex in curated[:10]] if curated else []
                integrations = self.dreamer.dream_integration(consolidated_facts, recent)

                if not integrations:
                    yield {"step": 6, "total": total_steps, "label": "REM: Generating", "status": "done",
                           "detail": "No integration data generated."}
                    rem_result = {"status": "skipped"}
                else:
                    rem_dir = Path(self.config.paths["training"]) / f"rem_{cycle_id}"
                    rem_dir.mkdir(parents=True, exist_ok=True)
                    rem_data = self.dreamer.dream_to_training_data(integrations, self.backend)
                    with open(rem_dir / "train.jsonl", "w") as f:
                        for item in rem_data:
                            f.write(json.dumps(item) + "\n")

                    yield {"step": 6, "total": total_steps, "label": "REM: Generating", "status": "done",
                           "detail": f"{len(integrations)} integration conversations"}

                    # [7] REM Train
                    yield {"step": 7, "total": total_steps, "label": "REM: Training", "status": "running"}
                    rem_adapter = self.trainer.train_rem(cycle_id, rem_dir)
                    yield {"step": 7, "total": total_steps, "label": "REM: Training", "status": "done",
                           "detail": f"Adapter: {rem_adapter}" if rem_adapter else "No data"}

                    # [8] REM Validate
                    yield {"step": 8, "total": total_steps, "label": "REM: Validating", "status": "running"}

                    rem_ppl = self.backend.compute_perplexity(ref_text) if ref_text else None
                    max_ppl_increase = self.config.get("rem.max_ppl_increase", 0.10)

                    ppl_ok = True
                    if sws_ppl and rem_ppl and sws_ppl > 0:
                        ppl_increase = (rem_ppl - sws_ppl) / sws_ppl
                        if ppl_increase > max_ppl_increase:
                            ppl_ok = False

                    recall_ok = True
                    sample = consolidated_facts[:5]
                    recalled = sum(1 for f in sample if self.memit_engine.test_recall(f)[0])
                    recall_rate = recalled / len(sample) if sample else 1.0
                    if recall_rate < 0.5:
                        recall_ok = False

                    if ppl_ok and recall_ok:
                        rem_detail = f"APPROVED. Recall: {recalled}/{len(sample)}"
                        if sws_ppl and rem_ppl:
                            rem_detail += f", PPL: {sws_ppl:.2f}→{rem_ppl:.2f}"
                        rem_result = {"status": "approved", "integrations": len(integrations)}
                    else:
                        reason = "PPL increase" if not ppl_ok else "Recall dropped"
                        rem_detail = f"REJECTED ({reason}). Rolled back to post-SWS."
                        latest = self.checkpoints.get_latest()
                        if latest:
                            self.backend.reload(latest["path"])
                        if self.memit_engine.enabled and hasattr(self.backend, "dequantize_layer"):
                            self.memit_engine._dequantize_target_layers()
                        if sws_snapshot:
                            self.memit_engine.restore_target_weights(sws_snapshot)
                        rem_result = {"status": "rejected", "reason": reason}

                    yield {"step": 8, "total": total_steps, "label": "REM: Validating", "status": "done",
                           "detail": rem_detail}

        # Report
        elapsed = time.time() - start_time
        report_step = total_steps
        summary_detail = f"facts_consolidated={facts_consolidated}"
        if rem_result:
            summary_detail += f", rem={rem_result.get('status', 'n/a')}"

        yield {"step": report_step, "total": total_steps, "label": "Summary", "status": "done",
               "detail": summary_detail, "facts_consolidated": facts_consolidated}
