"""Surprise estimator — computes novelty signal for consolidation gating.

Inspired by TITANS paper's surprise-gated memory. Computes a composite
surprise score from multiple signals to determine when buffered facts
should be consolidated into long-term MEMIT memory.
"""

import re
from typing import List, Optional


class SurpriseEstimator:
    """Computes a surprise score for each conversation turn.

    Surprise = weighted combination of:
      1. Fact novelty: fraction of extracted facts that are genuinely new
      2. Explicit markers: corrections, emphasis, personal revelations
      3. PPL-based: how surprising the user's input was to the model (optional)

    PPL component is off by default (costs a forward pass per turn).
    """

    def __init__(self, config, backend=None):
        cm_config = config.get("consolidation_moment", {}) or {}
        surprise_config = cm_config.get("surprise", {}) or {}

        self.consolidation_threshold = surprise_config.get("threshold", 0.6)

        self.novelty_weight = surprise_config.get("novelty_weight", 0.5)
        self.marker_weight = surprise_config.get("marker_weight", 0.3)
        self.ppl_weight = surprise_config.get("ppl_weight", 0.2)

        self.use_ppl = surprise_config.get("use_ppl", False)
        self._ppl_ema = None
        self._ppl_ema_alpha = surprise_config.get("ppl_ema_alpha", 0.3)

        self._backend = backend

    def evaluate(
        self,
        user_message: str,
        new_facts: list,
        total_extracted: int,
    ) -> float:
        """Compute composite surprise score for this turn.

        Args:
            user_message: The user's raw input
            new_facts: Facts that survived deduplication (genuinely new)
            total_extracted: Total facts extracted before deduplication

        Returns:
            Surprise score in [0.0, 1.0]
        """
        novelty = self._novelty_score(new_facts, total_extracted)
        markers = self._marker_score(user_message)

        ppl = 0.0
        if self.use_ppl and self._backend:
            ppl = self._ppl_surprise(user_message)

        # Weighted combination (normalize by active weights)
        total_weight = self.novelty_weight + self.marker_weight
        if self.use_ppl:
            total_weight += self.ppl_weight

        composite = (
            self.novelty_weight * novelty
            + self.marker_weight * markers
            + (self.ppl_weight * ppl if self.use_ppl else 0.0)
        ) / max(total_weight, 1e-6)

        return min(1.0, composite)

    def should_consolidate(self, surprise_score: float) -> bool:
        """Check if surprise exceeds consolidation threshold."""
        return surprise_score >= self.consolidation_threshold

    def _novelty_score(self, new_facts: list, total_extracted: int) -> float:
        """Fraction of extracted facts that are genuinely new.

        Soft ramp: 1 new fact = 0.5, 2 = 0.75, 3+ = 1.0.
        """
        if total_extracted == 0:
            return 0.0
        ratio = len(new_facts) / total_extracted
        count_boost = min(1.0, len(new_facts) / 2.0)
        return max(ratio, count_boost)

    def _marker_score(self, text: str) -> float:
        """Detect explicit memory-relevant markers in user input.

        Categories: corrections, emphasis, updates, personal revelations.
        """
        lower = text.lower()
        score = 0.0

        # Corrections (highest surprise — model's existing knowledge is wrong)
        correction_patterns = [
            r"\bactually\b", r"\bno,?\s+i\b", r"\bi meant\b",
            r"\bthat'?s (?:not |in)?correct\b", r"\bwrong\b",
        ]
        for pattern in correction_patterns:
            if re.search(pattern, lower):
                score = max(score, 0.9)
                break

        # Emphasis (user is signaling importance)
        emphasis_patterns = [
            r"\bremember (?:this|that)\b", r"\bimportant\b",
            r"\bdon'?t forget\b", r"\bmake sure (?:you |to )remember\b",
            r"\bi want you to (?:know|remember)\b",
        ]
        for pattern in emphasis_patterns:
            if re.search(pattern, lower):
                score = max(score, 0.8)
                break

        # Updates (something changed — old fact needs replacement)
        update_patterns = [
            r"\bi (?:just )?(?:changed|moved|switched|quit|left|started|got)\b",
            r"\bnot anymore\b", r"\bno longer\b", r"\bused to\b",
            r"\bi(?:'m| am) now\b",
        ]
        for pattern in update_patterns:
            if re.search(pattern, lower):
                score = max(score, 0.7)
                break

        # Personal revelations (novel personal info)
        revelation_patterns = [
            r"\bi(?:'ve| have) never (?:told|mentioned)\b",
            r"\bbetween (?:you and me|us)\b",
            r"\bmy (?:real|actual|full) name\b",
        ]
        for pattern in revelation_patterns:
            if re.search(pattern, lower):
                score = max(score, 0.7)
                break

        return score

    def _ppl_surprise(self, text: str) -> float:
        """Compute surprise from model perplexity on user's message.

        Uses exponential moving average: if this message's PPL is much
        higher than the running average, the input is surprising.
        """
        try:
            ppl = self._backend.compute_perplexity(text)
        except Exception:
            return 0.0

        if self._ppl_ema is None:
            self._ppl_ema = ppl
            return 0.0

        if self._ppl_ema > 0:
            ratio = ppl / self._ppl_ema
            surprise = max(0.0, min(1.0, (ratio - 1.0) * 2.0))
        else:
            surprise = 0.0

        # Update EMA
        self._ppl_ema = self._ppl_ema_alpha * ppl + (1 - self._ppl_ema_alpha) * self._ppl_ema

        return surprise

    def to_dict(self) -> dict:
        """Serializable status for API."""
        return {
            "threshold": self.consolidation_threshold,
            "weights": {
                "novelty": self.novelty_weight,
                "markers": self.marker_weight,
                "ppl": self.ppl_weight if self.use_ppl else 0.0,
            },
            "use_ppl": self.use_ppl,
            "ppl_ema": round(self._ppl_ema, 2) if self._ppl_ema else None,
        }
