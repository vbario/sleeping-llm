"""Valuation policies — pluggable priority scoring at the Tier-1 → Tier-2 choke point.

The selector is whatever sets QAPair.priority before facts are persisted to
the FactLedger. Policies implement the EGO-SELECT control ladder
(notes/131-ego-selector-experiment §4): random floor, surprise (production),
borrowed regex rules, LLM judge variants, full ego module, and oracle ceiling.

Contract: score_batch NEVER raises (chat.py swallows extraction exceptions;
that contract is preserved). Default policy 'surprise' with no turn metadata
is a passthrough — bit-identical to production behavior.
"""

import json
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EGO_PROMPTS_PATH = REPO_ROOT / "experiments" / "data" / "ego_prompts.yaml"
DEFAULT_RULES_PATH = REPO_ROOT / "experiments" / "data" / "borrowed_rules.yaml"

# Frozen output format (§4 "Valuation call format")
_OUTPUT_LINE_RE = re.compile(r"^F(\d+):\s*(\d{1,2})\s*$")

_FALLBACK_TEMPLATE = """{context_block}

{instruction}

Facts:
{fact_lines}

Answer with exactly one line per fact, in this exact format and nothing else:
F1: <integer 0-10>
F2: <integer 0-10>
...
"""

_FALLBACK_GENERIC_INSTRUCTION = (
    "For each fact: if this fact were permanently forgotten, "
    "how much future harm would result? 0-10."
)
_FALLBACK_SELF_INSTRUCTION = (
    "For each fact: if you permanently forgot this, how badly would your "
    "user, or your ability to serve them, be harmed? 0-10."
)

_prompts_cache = None


def _load_ego_prompts():
    """Load the frozen prompt artifacts (charter, filler, rules-prose, templates)."""
    global _prompts_cache
    if _prompts_cache is None:
        try:
            import yaml
            with open(EGO_PROMPTS_PATH) as f:
                _prompts_cache = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"  [Valuation] Could not load ego prompts ({e}), using inline fallbacks")
            _prompts_cache = {}
    return _prompts_cache


def _question_key(question):
    """Normalized question key (matches ledger dedup convention)."""
    return re.sub(r"\s+", " ", (question or "").lower().strip().rstrip("?.! "))


def _value_key(qa):
    """Normalized value for contradiction checks."""
    v = (getattr(qa, "value", "") or getattr(qa, "answer", "") or "")
    return re.sub(r"\s+", " ", v.lower().strip().rstrip(".,!? "))


class ValuationPolicy:
    """Base policy — passthrough scoring, never raises."""

    name = "base"

    def score_batch(self, qa_pairs, turn_meta, ledger_qas):
        """Score a consolidation batch. Returns list[float] in [0, 1].

        turn_meta is a list aligned with qa_pairs of per-fact dicts (or None).
        NEVER raises: any failure defaults the whole batch to 0.5.
        """
        if not qa_pairs:
            return []
        try:
            scores = self._score(qa_pairs, turn_meta, ledger_qas)
            return [min(1.0, max(0.0, float(s))) for s in scores]
        except Exception as e:
            print(f"  [Valuation] {self.name} scoring failed ({e}), defaulting to 0.5")
            return [0.5] * len(qa_pairs)

    def _score(self, qa_pairs, turn_meta, ledger_qas):
        return [getattr(qa, "priority", 0.5) for qa in qa_pairs]

    def rescore(self, ledger_entries, backend=None):
        """Optional pre-sleep re-scoring pass. None = unsupported."""
        return None

    def to_dict(self):
        return {"name": self.name}

    @staticmethod
    def _meta_at(turn_meta, i):
        if turn_meta and i < len(turn_meta):
            return turn_meta[i]
        return None


class RandomPolicy(ValuationPolicy):
    """C0 floor: seeded uniform random priority per fact."""

    name = "random"

    def __init__(self, seed=42):
        self.seed = seed
        self._rng = random.Random(seed)

    def _score(self, qa_pairs, turn_meta, ledger_qas):
        return [self._rng.uniform(0.0, 1.0) for _ in qa_pairs]

    def to_dict(self):
        return {"name": self.name, "seed": self.seed}


class SurprisePolicy(ValuationPolicy):
    """C1: the production selector.

    Without turn metadata this is a PASSTHROUGH returning each qa.priority
    unchanged (bit-identical production — chat.py already set it). With
    frozen-replay metadata ({'user_text', 'turn_new_facts'}) it recomputes
    evaluate(user_text, new_facts, len(new_facts)) exactly as chat.py:250
    does, including the novelty-degeneracy quirk (§2 scope 6).
    """

    name = "surprise"

    def __init__(self, estimator):
        self.estimator = estimator

    def _score(self, qa_pairs, turn_meta, ledger_qas):
        scores = []
        for i, qa in enumerate(qa_pairs):
            meta = self._meta_at(turn_meta, i)
            if not meta:
                scores.append(getattr(qa, "priority", 0.5))
                continue
            user_text = meta.get("user_text", "")
            n = max(1, int(meta.get("turn_new_facts", 1)))
            # Exact production call shape: evaluate(user_input, new_facts, len(new_facts)).
            # evaluate() only consumes len(new_facts), so a length-n list reproduces it.
            scores.append(self.estimator.evaluate(user_text, [qa] * n, n))
        return scores

    def to_dict(self):
        d = {"name": self.name}
        try:
            d["estimator"] = self.estimator.to_dict()
        except Exception:
            pass
        return d


class BorrowedPolicy(ValuationPolicy):
    """C2: static regex rule table (borrowed ego), first-match-wins."""

    name = "borrowed"

    def __init__(self, rules_yaml_path=None):
        import yaml
        path = Path(rules_yaml_path) if rules_yaml_path else DEFAULT_RULES_PATH
        if not path.is_absolute():
            path = REPO_ROOT / path
        self.rules_path = str(path)
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        self._rules = []
        self._default_score = 0.4
        for rule in data.get("rules", []):
            pattern = rule.get("pattern")
            if pattern is None:
                self._default_score = float(rule.get("score", 0.4))
            else:
                self._rules.append((
                    rule.get("name", "rule"),
                    re.compile(pattern, re.IGNORECASE),
                    float(rule.get("score", 0.4)),
                ))
        print(f"  [Valuation] borrowed: loaded {len(self._rules)} rule(s) "
              f"from {self.rules_path} (default {self._default_score})")

    def _score_one(self, qa):
        text = f"{qa.question} {qa.answer}"
        for _name, regex, score in self._rules:
            if regex.search(text):
                return score
        return self._default_score

    def _score(self, qa_pairs, turn_meta, ledger_qas):
        return [self._score_one(qa) for qa in qa_pairs]

    def to_dict(self):
        return {
            "name": self.name,
            "rules_path": self.rules_path,
            "rule_count": len(self._rules),
            "default_score": self._default_score,
        }


class JudgePolicy(ValuationPolicy):
    """C3/C3b/C4: one batched greedy LLM scoring call per consolidation moment.

    Prompt = context block + counterfactual instruction + facts listed as
    'F<n>: <statement>'. Output parsed as 'F<n>: <integer 0-10>' per line,
    clamped to [0, 10] / 10; unparsed lines fall back to 0.5 (per-line,
    counted). Full prompt, raw output, and token counts are logged (§7.2.2).
    """

    name = "judge"

    def __init__(self, backend, context_block, instruction, label="judge"):
        self.backend = backend
        self.context_block = context_block or ""
        self.instruction = instruction or _FALLBACK_GENERIC_INSTRUCTION
        self.name = label
        self.call_count = 0
        self.fallback_count = 0
        self.scored_count = 0
        self.last_fallbacks = 0
        lf = _load_ego_prompts().get("listing_format", {}) or {}
        self.template = lf.get("template", _FALLBACK_TEMPLATE)

    def _statement(self, qa):
        return (qa.answer or qa.question or "").strip()

    def _fact_line(self, idx, qa, meta):
        return f"F{idx}: {self._statement(qa)}"

    def _context(self):
        return self.context_block

    def _count_tokens(self, text):
        try:
            return self.backend.count_tokens(text)
        except Exception:
            return len(text.split())

    def _score(self, qa_pairs, turn_meta, ledger_qas):
        n = len(qa_pairs)
        fact_lines = "\n".join(
            self._fact_line(i + 1, qa, self._meta_at(turn_meta, i))
            for i, qa in enumerate(qa_pairs)
        )
        prompt = self.template.format(
            context_block=self._context(),
            instruction=self.instruction,
            fact_lines=fact_lines,
        )
        # Wrap in the chat template — raw text makes the instruct model
        # free-continue the prompt instead of answering (pilot §7.3a failure).
        chat_prompt = self.backend.apply_chat_template(
            [{"role": "user", "content": prompt}])
        raw = self.backend.generate(chat_prompt, max_tokens=8 * n + 24,
                                    temperature=0.0)
        self.call_count += 1
        print(f"  [Valuation] {self.name}: scored {n} fact(s) — "
              f"prompt_tokens={self._count_tokens(prompt)}, "
              f"output_tokens={self._count_tokens(raw)}")
        # §7.2.2: log every valuation call's FULL input
        print(f"  [Valuation] {self.name} full prompt:\n{prompt}")
        print(f"  [Valuation] {self.name} raw output:\n{raw}")

        scores, fallbacks = self._parse(raw, n)
        self.fallback_count += fallbacks
        self.scored_count += n
        self.last_fallbacks = fallbacks
        if fallbacks:
            print(f"  [Valuation] {self.name}: {fallbacks}/{n} line(s) fell back to 0.5")
        return scores

    @staticmethod
    def _parse(raw, n):
        """Parse 'F<n>: <int>' lines. Returns (scores, fallback_count)."""
        by_index = {}
        for line in (raw or "").splitlines():
            m = _OUTPUT_LINE_RE.match(line.strip())
            if not m:
                continue
            idx = int(m.group(1))
            if 1 <= idx <= n and idx not in by_index:
                by_index[idx] = min(10, max(0, int(m.group(2)))) / 10.0
        scores = []
        fallbacks = 0
        for i in range(1, n + 1):
            if i in by_index:
                scores.append(by_index[i])
            else:
                scores.append(0.5)
                fallbacks += 1
        return scores, fallbacks

    def to_dict(self):
        return {
            "name": self.name,
            "call_count": self.call_count,
            "fallback_count": self.fallback_count,
            "scored_count": self.scored_count,
            "last_fallbacks": self.last_fallbacks,
            "instruction": self.instruction,
        }


class EgoFullPolicy(JudgePolicy):
    """C5: judge scoring against the evolving self-model, plus per-fact
    [t, source, prov] metadata, pre-sleep rescore, and supersession detection.
    """

    name = "ego_full"
    MAX_RESCORE = 8

    def __init__(self, backend, self_model=None, context_block=None, instruction=None):
        if instruction is None:
            instruction = _load_ego_prompts().get(
                "instruction_self_indexed", _FALLBACK_SELF_INSTRUCTION)
        super().__init__(backend, context_block, instruction, label="ego_full")
        self.self_model = self_model
        self._supersessions = []

    def _context(self):
        """Evolving self-model block; falls back to the seeded charter."""
        if self.self_model is not None:
            try:
                return self.self_model.get_prompt_block()
            except Exception as e:
                print(f"  [Valuation] ego_full: self-model unavailable ({e}), using static block")
        return self.context_block

    def _fact_line(self, idx, qa, meta):
        meta = meta or {}
        session = meta.get("session", "?")
        prov = getattr(qa, "provenance", "user_stated") or "user_stated"
        speaker = meta.get("speaker") or (
            "assistant" if prov == "assistant_generated" else "user")
        return (f"F{idx}: {self._statement(qa)} "
                f"[t=session {session}, source={speaker}, prov={prov}]")

    def _score(self, qa_pairs, turn_meta, ledger_qas):
        self._supersessions = self._find_supersessions(qa_pairs, ledger_qas)
        return super()._score(qa_pairs, turn_meta, ledger_qas)

    def _find_supersessions(self, qa_pairs, ledger_qas):
        """Ledger questions contradicted by the incoming batch (§4 C5(v)).

        Heuristic: same normalized question key, different normalized value.
        """
        batch = {}
        for qa in qa_pairs:
            key = _question_key(qa.question)
            if key:
                batch[key] = _value_key(qa)
        hits = []
        for lq in (ledger_qas or []):
            key = _question_key(lq.question)
            if key in batch and batch[key] != _value_key(lq):
                hits.append(lq.question)
        if hits:
            print(f"  [Valuation] ego_full: {len(hits)} superseded ledger fact(s) detected")
        return hits

    def supersessions(self):
        """Ledger questions contradicted by the most recently scored batch."""
        return list(self._supersessions)

    def rescore(self, ledger_entries, backend=None):
        """Pre-sleep re-scoring of ≤8 active ledger entries.

        Returns priorities aligned with ledger_entries; entries not rescored
        (pruned, or beyond the cap) keep their current priority. Per-entry
        [t, source, prov] metadata is rebuilt from the ledger entry
        (arrival_session stamped at the ledger-merge step; QAPair carries
        provenance — §4 C5(i)).
        """
        from src.memory.facts import QAPair

        entries = ledger_entries or []
        out = [e.get("qa", {}).get("priority", 0.5) for e in entries]
        active_idx = [i for i, e in enumerate(entries)
                      if not e.get("pruned", False)][:self.MAX_RESCORE]
        if not active_idx:
            return out

        qa_pairs = [QAPair.from_dict(entries[i]["qa"]) for i in active_idx]
        metas = []
        for i in active_idx:
            e = entries[i]
            prov = (e.get("qa", {}) or {}).get("provenance", "user_stated")
            metas.append({
                "session": e.get("arrival_session", "?"),
                "speaker": "assistant" if prov == "assistant_generated" else "user",
            })
        print(f"  [Valuation] ego_full rescore: {len(qa_pairs)} active ledger entr(ies)")
        old_backend = self.backend
        if backend is not None:
            self.backend = backend
        try:
            scores = self.score_batch(qa_pairs, metas, None)
        finally:
            self.backend = old_backend
        for j, i in enumerate(active_idx):
            out[i] = scores[j]
        return out

    def to_dict(self):
        d = super().to_dict()
        d["has_self_model"] = self.self_model is not None
        d["last_supersessions"] = list(self._supersessions)
        return d


class OraclePolicy(ValuationPolicy):
    """C6 ceiling: ground-truth value labels from the corpus.

    Labels keyed by normalized question; each label carries
    (value_pre, value_post). Post-shift labels apply from shift_session on
    (meta carries the session number). Unknown facts default to 0.5.
    """

    name = "oracle"

    def __init__(self, labels_by_question, shift_session=8):
        self.shift_session = shift_session
        self._labels = {}
        for key, val in (labels_by_question or {}).items():
            self._labels[_question_key(key)] = self._norm_label(val)
        self.miss_count = 0

    @staticmethod
    def _norm_label(val):
        """Accept scalar, [pre, post], or {'value_pre', 'value_post'} labels."""
        if isinstance(val, dict):
            pre = float(val.get("value_pre", val.get("value", 0.5)))
            post = float(val.get("value_post", pre))
        elif isinstance(val, (list, tuple)) and val:
            pre, post = float(val[0]), float(val[-1])
        else:
            pre = post = float(val)
        return (pre, post)

    def _score(self, qa_pairs, turn_meta, ledger_qas):
        scores = []
        for i, qa in enumerate(qa_pairs):
            meta = self._meta_at(turn_meta, i) or {}
            session = int(meta.get("session", 0))
            label = self._labels.get(_question_key(qa.question))
            if label is None:
                self.miss_count += 1
                print(f"  [Valuation] oracle: no label for '{qa.question}', defaulting to 0.5")
                scores.append(0.5)
            else:
                pre, post = label
                scores.append(post if session >= self.shift_session else pre)
        return scores

    def to_dict(self):
        return {
            "name": self.name,
            "label_count": len(self._labels),
            "shift_session": self.shift_session,
            "miss_count": self.miss_count,
        }


def build_policy(config, backend, self_model=None):
    """Construct the policy named by config valuation.policy.

    Returns None (no policy installed → production behavior) on unknown
    names or construction failure.
    """
    val = config.get("valuation", {}) or {}
    name = (val.get("policy") or "surprise").lower()

    try:
        if name == "surprise":
            from src.wake.surprise import SurpriseEstimator
            return SurprisePolicy(SurpriseEstimator(config, backend))

        if name == "random":
            return RandomPolicy(seed=int(val.get("seed", 42)))

        if name == "borrowed":
            return BorrowedPolicy(val.get("rules_file"))

        prompts = _load_ego_prompts()
        generic = (prompts.get("instruction_generic")
                   or _FALLBACK_GENERIC_INSTRUCTION).strip()
        self_indexed = (prompts.get("instruction_self_indexed")
                        or _FALLBACK_SELF_INSTRUCTION).strip()
        charter = (prompts.get("charter_seed") or "").strip()

        if name == "judge":
            return JudgePolicy(backend, (prompts.get("filler_generic") or "").strip(),
                               generic, label="judge")

        if name == "judge_rules":
            return JudgePolicy(backend, (prompts.get("rules_prose") or "").strip(),
                               generic, label="judge_rules")

        if name == "ego_static":
            context = charter
            if not context and self_model is not None:
                context = self_model.get_prompt_block()
            return JudgePolicy(backend, context, self_indexed, label="ego_static")

        if name == "ego_full":
            return EgoFullPolicy(backend, self_model,
                                 context_block=charter, instruction=self_indexed)

        if name == "oracle":
            labels_path = val.get("oracle_labels")
            if not labels_path:
                print("  [Valuation] oracle policy needs valuation.oracle_labels — no policy installed")
                return None
            path = Path(labels_path)
            if not path.is_absolute():
                path = REPO_ROOT / path
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict) and "labels" in data:
                labels = data["labels"]
                shift_session = int(data.get("shift_session", 8))
            else:
                labels = data
                shift_session = 8
            return OraclePolicy(labels, shift_session=shift_session)

        print(f"  [Valuation] Unknown policy '{name}' — no policy installed")
        return None
    except Exception as e:
        print(f"  [Valuation] Failed to build policy '{name}' ({e}) — no policy installed")
        return None
