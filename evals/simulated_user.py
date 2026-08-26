"""An LLM-backed fake user for the eval harness.

Answers Keel's questions in character: plainly, without volunteering unasked
information, and occasionally declining. Falls back to a deterministic "mostly
accept defaults" strategy when no API key is available, so the harness still runs
(with reduced realism) offline.
"""
from __future__ import annotations

import random

from keel.llm import complete

SYSTEM_TEMPLATE = """You are role-playing as a software developer being interviewed by a CLI \
tool that turns a vague project idea into a structured prompt for a coding agent.

Persona: {persona}
Original request you typed: "{prompt}"

You will be shown one question at a time, with a recommended default. Answer as this \
persona would, in ONE short sentence or phrase -- plain and direct, no filler.

Rules:
- If the recommended default already matches what you want, respond with exactly: accept
- Never volunteer information about slots you have not been asked about yet.
- Occasionally, if you genuinely have no strong opinion on this particular question, \
respond with exactly: skip
Respond with ONLY your answer text, or the literal word "accept", or the literal word "skip"."""


class SimulatedUser:
    def __init__(self, persona: str, prompt: str, decline_rate: float = 0.1, seed: int | None = None):
        self.persona = persona
        self.prompt = prompt
        self.decline_rate = decline_rate
        self._rng = random.Random(seed)

    def answer(self, question: str, default: str) -> str:
        """Returns raw input as Keel's apply_answer() expects: "" to accept, "skip" to
        decline, or free text."""
        result = complete(
            SYSTEM_TEMPLATE.format(persona=self.persona, prompt=self.prompt),
            f"Question: {question}\nRecommended default: {default}\nYour answer:",
            max_tokens=120,
            effort="low",
        )

        if result is None:
            if self._rng.random() < self.decline_rate:
                return "skip"
            return ""

        text = result.strip()
        if text.lower() == "accept":
            return ""
        if text.lower() == "skip":
            return "skip"
        return text
