from __future__ import annotations

import json
from typing import Any

from autoflow.llm.client import LLMClient


SCRIPT_AUTHOR_SYSTEM_PROMPT = """You write constrained Python scripts for AutoFlow authorized security testing.
Return only a compact JSON object. Do not include markdown.
The script must target only the provided target and stay within the provided policy profile.
Do not include persistence, destructive writes, evasion, reverse connections, credential theft, or out-of-scope access.
Use short network timeouts and print a concise JSON object or concise text.
"""


class ScriptAuthor:
    """Generate script drafts for Executor; policy and Docker sandbox still decide execution."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def author(self, context: dict[str, Any]) -> str:
        return self._request_script(context=context, repair_context=None)

    def repair(self, context: dict[str, Any], failed_script: str, failure: dict[str, Any]) -> str:
        repair_context = {
            "failed_script": failed_script,
            "failure": failure,
            "instruction": "Repair the script while staying inside the same policy profile.",
        }
        return self._request_script(context=context, repair_context=repair_context)

    def _request_script(
        self,
        context: dict[str, Any],
        repair_context: dict[str, Any] | None,
    ) -> str:
        client = self.llm_client or LLMClient()
        policy_profile = context.get("metadata", {}).get("script_policy_profile", "low_readonly_http")
        prompt = {
            "task": "Generate a constrained Python script for this TestPlanAction.",
            "target": context.get("target"),
            "target_scope": context.get("target_scope", []),
            "strategy": context.get("type"),
            "name": context.get("name"),
            "rationale": context.get("rationale"),
            "expected_impact": context.get("expected_impact"),
            "metadata": context.get("metadata", {}),
            "script_goal": context.get("metadata", {}).get("script_goal"),
            "policy_profile": policy_profile,
            "repair_context": repair_context,
            "requirements": {
                "language": "python3",
                "respect_policy_profile": True,
                "max_default_timeout_seconds": 10,
                "target_must_remain_in_scope": True,
                "output_schema": "print one JSON object when possible",
            },
            "output_schema": {
                "script_source": "complete Python script as a string",
            },
        }
        response = client.complete_json(
            prompt=json.dumps(prompt, ensure_ascii=False),
            system=SCRIPT_AUTHOR_SYSTEM_PROMPT,
            max_tokens=1600,
        )
        script_source = response.get("script_source", "")
        if not isinstance(script_source, str) or not script_source.strip():
            raise ValueError("ScriptAuthor response did not include script_source")
        return script_source.strip()
