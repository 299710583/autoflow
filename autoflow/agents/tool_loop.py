from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from autoflow.graph.state import AutoFlowState
from autoflow.llm.client import LLMClient, parse_json_object
from autoflow.tools.catalog import ToolCatalog
from autoflow.tools.dispatcher import ToolDispatcher


@dataclass
class ToolLoopResult:
    final: dict[str, Any]
    messages: list[dict[str, Any]]
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0


class AgentToolLoop:
    """Reusable OpenAI-compatible function-calling loop for AutoFlow agents."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        catalog: ToolCatalog | None = None,
        dispatcher: ToolDispatcher | None = None,
        max_tool_rounds: int = 8,
        max_tool_calls: int = 16,
        max_tokens: int = 4096,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.catalog = catalog or ToolCatalog()
        self.dispatcher = dispatcher or ToolDispatcher(catalog=self.catalog)
        self.max_tool_rounds = max_tool_rounds
        self.max_tool_calls = max_tool_calls
        self.max_tokens = max_tokens

    def run(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        state: AutoFlowState,
        final_repair_instruction: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> ToolLoopResult:
        user_payload = self._with_tool_manifest(user_payload)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        tool_schemas = tools or self.catalog.openai_tools()
        tool_results: list[dict[str, Any]] = []
        tool_call_count = 0

        for iteration in range(1, self.max_tool_rounds + 1):
            assistant_message = self.llm_client.chat_with_tools(
                messages=messages,
                tools=tool_schemas,
                max_tokens=self.max_tokens,
            )
            messages.append(self._assistant_message_for_history(assistant_message))
            tool_calls = assistant_message.get("tool_calls") or []
            if tool_calls:
                for call in tool_calls:
                    if tool_call_count >= self.max_tool_calls:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.get("id"),
                                "content": json.dumps(
                                    {
                                        "ok": False,
                                        "error": (
                                            "Tool-call budget exhausted. Stop calling tools and produce final JSON "
                                            "from the observations already available."
                                        ),
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        )
                        continue
                    result = self._dispatch_call(call, state)
                    tool_call_count += 1
                    tool_results.append(result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id"),
                            "content": self.dispatcher.tool_message_content(result),
                        }
                    )
                continue

            content = assistant_message.get("content", "")
            try:
                final = parse_json_object(content)
            except (json.JSONDecodeError, ValueError) as exc:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response was not valid final JSON: {exc}. "
                            f"{final_repair_instruction} Return exactly one JSON object, no markdown."
                        ),
                    }
                )
                continue
            return ToolLoopResult(final=final, messages=messages, tool_results=tool_results, iterations=iteration)

        messages.append(
            {
                "role": "user",
                "content": (
                    "The tool-call loop reached its maximum number of rounds. "
                    f"{final_repair_instruction} Return exactly one final JSON object now."
                ),
            }
        )
        final: dict[str, Any] | None = None
        for _ in range(3):
            assistant_message = self.llm_client.chat_with_tools(
                messages=messages,
                tools=tool_schemas,
                max_tokens=self.max_tokens,
                tool_choice="none",
            )
            messages.append(self._assistant_message_for_history(assistant_message))
            try:
                final = parse_json_object(assistant_message.get("content", ""))
                break
            except (json.JSONDecodeError, ValueError) as exc:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous final response was not valid JSON: {exc}. "
                            f"{final_repair_instruction} Return exactly one JSON object, no markdown."
                        ),
                    }
                )
        if final is None:
            final = parse_json_object(messages[-2].get("content", ""))
        return ToolLoopResult(
            final=final,
            messages=messages,
            tool_results=tool_results,
            iterations=self.max_tool_rounds,
        )

    def _dispatch_call(self, call: dict[str, Any], state: AutoFlowState) -> dict[str, Any]:
        function = call.get("function") or {}
        name = str(function.get("name", ""))
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return {
                "ok": False,
                "tool_call": name,
                "error": f"Tool arguments were not valid JSON: {exc}",
                "raw_arguments": raw_arguments,
            }
        if not isinstance(arguments, dict):
            return {
                "ok": False,
                "tool_call": name,
                "error": "Tool arguments must be a JSON object.",
                "raw_arguments": raw_arguments,
            }
        return self.dispatcher.dispatch(name, arguments, state)

    def _assistant_message_for_history(self, message: dict[str, Any]) -> dict[str, Any]:
        history = {"role": "assistant", "content": message.get("content", "")}
        if message.get("tool_calls"):
            history["tool_calls"] = message["tool_calls"]
        return history

    def _with_tool_manifest(self, user_payload: dict[str, Any]) -> dict[str, Any]:
        if "available_tool_manifest" in user_payload:
            return user_payload
        manifest = getattr(self.catalog, "manifest", None)
        prompt_manifest = getattr(manifest, "prompt_manifest", None)
        if not callable(prompt_manifest):
            return user_payload
        enriched = dict(user_payload)
        enriched["available_tool_manifest"] = prompt_manifest(None)
        enriched["tool_execution_boundary"] = {
            "containerized": True,
            "container_image": "autoflow-kali-tools",
            "host_shell_available_to_llm": False,
            "source_audit_paths": ["data/artifacts", "data/source", "data/source_audit"],
        }
        return enriched
