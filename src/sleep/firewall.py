"""Hallucination firewall — verifies extracted facts against source conversation.

After the curator extracts Q&A pairs using the model, this module checks
each pair against the original conversation text. Pairs containing claims
not grounded in the source conversation are discarded before training.
"""

import re


class HallucinationFirewall:
    """Verifies Q&A training pairs against the source conversation."""

    def __init__(self, config, backend=None):
        self.config = config
        self.backend = backend
        firewall_config = config.sleep.get("firewall", {})
        self.min_grounding_score = firewall_config.get("min_grounding_score", 0.5)
        self.use_model_verification = firewall_config.get("use_model_verification", False)

    def verify_pairs(self, qa_pairs, conversation_text):
        """Verify Q&A pairs against the source conversation.

        Args:
            qa_pairs: List of [user_msg_dict, assistant_msg_dict] pairs
            conversation_text: The raw conversation text these were extracted from

        Returns:
            Tuple of (verified_pairs, rejected_pairs)
        """
        verified = []
        rejected = []

        conv_lower = conversation_text.lower()

        for pair in qa_pairs:
            answer = pair[1]["content"]

            # Pass 1: Clean extraction artifacts
            cleaned = self._clean_answer(answer)
            if cleaned != answer:
                pair[1]["content"] = cleaned
                answer = cleaned

            # Pass 2: Entity grounding check
            grounding_score = self._check_grounding(answer, conv_lower)

            # Pass 3: Optional model verification for borderline cases
            if (self.use_model_verification and self.backend
                    and 0.3 <= grounding_score < self.min_grounding_score):
                if self._model_verify(pair, conversation_text):
                    grounding_score = self.min_grounding_score

            if grounding_score >= self.min_grounding_score:
                verified.append(pair)
            else:
                rejected.append({
                    "pair": pair,
                    "grounding_score": grounding_score,
                })

        return verified, rejected

    def _clean_answer(self, answer):
        """Remove extraction artifacts from answers.

        The model often appends numbered list items from subsequent facts.
        E.g.: "Andre Patandre  2. Andre Patandre is a music producer."
        Should become: "Andre Patandre"
        """
        # Remove trailing numbered list items: digits + period + space
        cleaned = re.split(r'\s+\d+\.\s', answer)[0].strip()

        # Remove "Note that..." disclaimers
        cleaned = re.split(r'\s*Note that\s', cleaned, flags=re.IGNORECASE)[0].strip()

        # Remove "Note:" disclaimers
        cleaned = re.split(r'\s*Note:\s', cleaned, flags=re.IGNORECASE)[0].strip()

        return cleaned if cleaned else answer

    def _check_grounding(self, answer, conv_lower):
        """Check what fraction of key claims in the answer appear in the conversation.

        Returns a score from 0.0 (no grounding) to 1.0 (fully grounded).
        """
        claims = self._extract_claims(answer)

        if not claims:
            return 1.0  # nothing specific to check

        grounded = 0
        for claim in claims:
            if claim.lower() in conv_lower:
                grounded += 1

        return grounded / len(claims)

    def _extract_claims(self, text):
        """Extract verifiable claims/entities from text.

        Focuses on proper nouns, numbers, multi-word names, and quoted strings.
        """
        claims = []
        words = text.split()

        for i, word in enumerate(words):
            clean = word.strip('.,!?";:()[]\'')
            if not clean:
                continue

            # Capitalized words (not sentence-initial) are likely proper nouns
            if clean[0].isupper() and i > 0 and not words[i - 1].endswith('.'):
                claims.append(clean)

            # Numbers are specific claims
            if any(c.isdigit() for c in clean) and len(clean) <= 10:
                claims.append(clean)

        # Multi-word proper nouns (consecutive capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        claims.extend(proper_nouns)

        # Quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        claims.extend(quoted)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in claims:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique.append(c)

        return unique

    def _model_verify(self, qa_pair, conversation_text):
        """Use the model to verify if a fact is grounded in the conversation."""
        answer = qa_pair[1]["content"]

        verify_prompt = [
            {
                "role": "user",
                "content": (
                    "Does the following claim appear in or follow directly from "
                    "the conversation below? Answer only YES or NO.\n\n"
                    f"Claim: {answer}\n\n"
                    f"Conversation:\n{conversation_text[:2000]}\n\n"
                    "Answer (YES or NO):"
                ),
            }
        ]

        prompt = self.backend.apply_chat_template(verify_prompt)
        response = self.backend.generate(prompt, max_tokens=5, temperature=0.1)

        return "yes" in response.strip().lower()
