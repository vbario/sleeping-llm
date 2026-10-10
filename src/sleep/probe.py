"""Recall probe — asks a fact's held-out phrasings with the fact withheld.

Shared by graduation tests (full sleep), audits (nap, regression gate) and
experiments, so every recall decision is made the same way:

  * the fact under test is removed from the "Things you remember" list,
    every other active fact stays (mirrors real conditions);
  * the held-out paraphrases are asked, never the trained phrasing
    (falls back to the canonical question only when a fact has no held-out
    set, and says so in the result);
  * each response is scored with recall_match (word boundary, unhedged);
  * negative values (an old, superseded value) must NOT appear.
"""

from typing import List, Sequence

from src.memory.facts import QAPair
from src.sleep.recall_match import normalize, value_in_response, first_value_in_response


class RecallProbe:
    def __init__(self, backend, system_prompt: str, temperature: float = 0.1,
                 max_tokens: int = 100, min_pass_fraction: float = 1.0):
        self.backend = backend
        self.system_prompt = system_prompt or ""
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_pass_fraction = min_pass_fraction

    @staticmethod
    def others(qa: QAPair, all_qa: Sequence[QAPair]) -> List[QAPair]:
        """Every fact except the one under test (matched by question or answer)."""
        q = normalize(qa.question)
        a = normalize(qa.answer)
        return [f for f in all_qa
                if normalize(f.question) != q and normalize(f.answer) != a]

    def build_system(self, other_qa: Sequence[QAPair]) -> str:
        parts = [self.system_prompt]
        if other_qa:
            lines = [f"- {f.answer}" for f in other_qa]
            parts.append("Things you remember about the user:\n" + "\n".join(lines))
        return "\n\n".join(p for p in parts if p)

    def ask(self, question: str, other_qa: Sequence[QAPair]) -> str:
        messages = [
            {"role": "system", "content": self.build_system(other_qa)},
            {"role": "user", "content": question},
        ]
        prompt = self.backend.apply_chat_template(messages)
        return self.backend.generate(prompt, max_tokens=self.max_tokens,
                                     temperature=self.temperature)

    def test(self, qa: QAPair, all_qa: Sequence[QAPair],
             negative_values: Sequence[str] = ()) -> dict:
        """Probe one fact. Returns per-phrasing outcomes and the overall verdict."""
        other_qa = self.others(qa, all_qa)
        phrasings = list(qa.held_out) if qa.held_out else [qa.question]
        fallback = not bool(qa.held_out)

        results = []
        negative_hit = None
        for phrasing in phrasings:
            response = self.ask(phrasing, other_qa)
            ok = value_in_response(qa.value, response)
            neg = first_value_in_response(negative_values, response) if negative_values else None
            if neg and not ok:
                negative_hit = neg
            elif neg and ok and response.lower().find(neg.lower()) < response.lower().find(qa.value.lower()):
                negative_hit = neg
            results.append({"question": phrasing, "response": response, "passed": ok,
                            "negative": neg})

        passed_n = sum(1 for r in results if r["passed"])
        fraction = passed_n / len(results) if results else 0.0
        return {
            "passed": fraction >= self.min_pass_fraction and negative_hit is None,
            "fraction": round(fraction, 3),
            "phrasings": results,
            "fallback": fallback,
            "negative_hit": negative_hit,
        }
