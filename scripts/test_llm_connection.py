"""Test the configured OpenAI-compatible LLM endpoint."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoflow.llm.client import LLMClient
from autoflow.settings import settings


def main() -> None:
    print(f"Testing LLM: model={settings.llm_model} base_url={settings.llm_base_url}")
    try:
        result = LLMClient().ping()
    except Exception as exc:
        print(f"FAILED: {exc}")
        raise SystemExit(1) from exc

    print(result)


if __name__ == "__main__":
    main()
