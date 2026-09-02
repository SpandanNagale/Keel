"""A stand-in for the developer answering Keel's questions.

Two modes, chosen by whether a provider is available:

* **llm** — an LLM plays the persona: answers the question plainly in one or two
  sentences, volunteers nothing beyond what was asked, and says "not sure" (which
  the harness turns into an accept-the-recommendation) when the persona would not
  know. This is the realistic mode.
* **deterministic** — no LLM: accept the recommended answer every time, and skip
  a low-priority optional slot on a fixed schedule. Reproducible; used when no
  provider is configured so ``run_evals.py`` still runs.

Either way the return is ``(action, text)`` where ``action`` is ``"answer"``,
``"accept"``, or ``"skip"``.
"""
from __future__ import annotations

from keel import llm

_PERSONA_SYSTEM = """You are role-playing a software developer who had a vague idea and is now \
being asked clarifying questions about it. Stay in character.

Rules:
- Answer the ONE question asked, in at most two plain sentences. Concrete, no hedging prose.
- Volunteer nothing the question did not ask for.
- If the persona genuinely would not have thought about this, reply with exactly: NOT SURE
- Never invent specific numbers, product names, or versions the persona was not given.

Respond with a JSON object: {"answer": "..."}  (or {"answer": "NOT SURE"})."""


class SimulatedUser:
    def __init__(self, persona: str, provider: "llm.Provider | None", *, seed: int = 0):
        self.persona = persona
        self.provider = provider
        self._n = seed

    @property
    def mode(self) -> str:
        return "llm" if self.provider is not None else "deterministic"

    def respond(
        self, question: str, recommended: str, established: list[str]
    ) -> tuple[str, str]:
        self._n += 1
        if self.provider is None:
            # Accept most; skip every 4th question (stands in for "sometimes skips").
            if self._n % 4 == 0:
                return "skip", ""
            return "accept", recommended

        user = (
            f"Persona: {self.persona}\n\n"
            + ("Already answered:\n" + "\n".join(f"- {e}" for e in established) + "\n\n"
               if established else "")
            + f'Question: "{question}"\n\n'
            f'A suggested answer you may adopt if it fits: "{recommended}"\n\n'
            "Give the persona's answer as JSON."
        )
        result, error = llm.complete_json(_PERSONA_SYSTEM, user, provider=self.provider,
                                          max_tokens=250)
        if error is not None or not isinstance(result, dict):
            return "accept", recommended
        answer = str(result.get("answer", "")).strip()
        if not answer or answer.upper().startswith("NOT SURE"):
            return "accept", recommended
        return "answer", answer
