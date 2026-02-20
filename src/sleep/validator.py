"""Validator — evaluates model quality before and after sleep to detect drift."""

import json
from pathlib import Path


class SleepValidator:
    """Runs benchmark evaluations to detect model degradation.

    If post-sleep scores drop below a threshold compared to pre-sleep,
    the sleep merge is blocked and the model rolls back.
    """

    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        self.benchmark_dir = Path(config.paths["benchmarks"])
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.min_score_ratio = config.validation["min_score_ratio"]
        self.num_questions = config.validation["num_questions"]

    def evaluate(self):
        """Run benchmark evaluation and return a score.

        Returns:
            dict with 'score' (0-1), 'total', 'correct', 'results' details
        """
        questions = self._load_benchmark_questions()
        if not questions:
            return {"score": 1.0, "total": 0, "correct": 0, "results": []}

        questions = questions[:self.num_questions]
        correct = 0
        results = []

        for q in questions:
            prompt_messages = [{"role": "user", "content": q["question"]}]
            prompt = self.backend.apply_chat_template(prompt_messages)
            response = self.backend.generate(prompt, max_tokens=100, temperature=0.1)

            passed = self._check_answer(response, q["expected_keywords"])
            if passed:
                correct += 1

            results.append({
                "question": q["question"],
                "response": response,
                "expected": q["expected_keywords"],
                "passed": passed,
            })

        score = correct / len(questions) if questions else 1.0
        return {
            "score": score,
            "total": len(questions),
            "correct": correct,
            "results": results,
        }

    def validate_sleep(self, pre_score, post_score):
        """Compare pre/post sleep scores and decide if merge is safe.

        Args:
            pre_score: Score dict from before sleep training
            post_score: Score dict from after sleep training

        Returns:
            dict with 'approved' bool and 'reason' string
        """
        if pre_score["total"] == 0:
            return {"approved": True, "reason": "No benchmarks configured, auto-approve"}

        pre = pre_score["score"]
        post = post_score["score"]

        if pre == 0:
            return {"approved": True, "reason": "Pre-sleep score was 0, nothing to compare"}

        ratio = post / pre

        if ratio >= self.min_score_ratio:
            return {
                "approved": True,
                "reason": f"Score ratio {ratio:.2f} >= threshold {self.min_score_ratio}",
                "pre_score": pre,
                "post_score": post,
            }
        else:
            return {
                "approved": False,
                "reason": (
                    f"Score dropped: {pre:.2f} -> {post:.2f} "
                    f"(ratio {ratio:.2f} < threshold {self.min_score_ratio})"
                ),
                "pre_score": pre,
                "post_score": post,
            }

    def _check_answer(self, response, expected_keywords):
        """Check if a response contains expected keywords."""
        response_lower = response.lower()
        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                return True
        return False

    def _load_benchmark_questions(self):
        """Load benchmark questions from the benchmarks directory."""
        benchmark_file = self.benchmark_dir / "questions.jsonl"
        if not benchmark_file.exists():
            return []
        questions = []
        with open(benchmark_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
        return questions
