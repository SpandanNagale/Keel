"""Keel smoke test: make exactly one live LLM call and report what happened.

This is the 30-second diagnosis the previous build lacked. Run it before wiring
or deploying the UI to confirm the model actually answers.

    python doctor.py                          # provider from env / secrets.toml
    KEEL_PROVIDER=ollama python doctor.py     # local Ollama daemon, no key
    python doctor.py --provider groq          # force a hosted provider
    python doctor.py --model openai/gpt-oss-120b

Hosted provider order when several keys are present: Groq, Ollama Cloud, then
Anthropic. Exit 0 = a parseable JSON object came back; non-zero = it did not and
the raw reason is printed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from keel import llm


def _secrets_map() -> dict[str, str]:
    secrets = Path(__file__).parent / ".streamlit" / "secrets.toml"
    if not secrets.is_file():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - py311+ has tomllib
        return {}
    try:
        data = tomllib.loads(secrets.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: str(v) for k, v in data.items() if isinstance(v, (str, int, float))}


def main() -> int:
    parser = argparse.ArgumentParser(description="One live Keel LLM call.")
    parser.add_argument("--provider", default=None,
                        choices=[*llm.PROVIDER_ORDER, "ollama"],
                        help="Force a provider ('ollama' = local daemon).")
    parser.add_argument("--key", default=None, help="Hosted API key (with --provider).")
    parser.add_argument("--model", default=None, help="Model string (default: provider default).")
    args = parser.parse_args()

    available = {**_secrets_map(),
                 **{k: v for k, v in os.environ.items() if k in llm.SECRET_KEYS.values()}}
    env = dict(os.environ)
    if args.provider:
        env["KEEL_PROVIDER"] = args.provider
    if args.key and args.provider and args.provider in llm.SECRET_KEYS:
        available[llm.SECRET_KEYS[args.provider]] = args.key

    provider, reason = llm.resolve_provider(available, model_override=args.model, env=env)
    if provider is None:
        print(f"No usable provider: {reason}")
        return 2

    print(f"Provider: {provider.name}")
    print(f"Model:    {provider.model}")
    if provider.host:
        print(f"Host:     {provider.host}")
    if provider.api_key:
        print(f"Key:      ...{provider.api_key[-4:]} ({len(provider.api_key)} chars)")
    else:
        print("Key:      (none — local daemon)")
    print("Calling the model with a trivial JSON task...")

    result, error = llm.complete_json(
        "Respond with ONLY a JSON object. No prose, no code fences.",
        'Return exactly this object: {"ok": true, "keel": "healthy"}',
        provider=provider,
    )

    if error is not None:
        print(f"\nFAILED: {error}")
        return 1

    print(f"\nOK. Parsed response: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
