from __future__ import annotations

import json
import unittest

from autoflow.agents.tool_loop import AgentToolLoop


class FakeToolLLM:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def chat_with_tools(self, messages, tools, max_tokens=1024, tool_choice="auto"):
        self.calls.append([dict(message) for message in messages])
        if len(self.calls) == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "web_recon_fetch_page",
                            "arguments": json.dumps({"target": "http://example.test"}),
                        },
                    }
                ],
                "finish_reason": "tool_calls",
            }
        return {
            "role": "assistant",
            "content": json.dumps({"attack_surfaces": [], "test_plans": []}),
            "tool_calls": [],
            "finish_reason": "stop",
        }


class FakeDispatcher:
    def __init__(self) -> None:
        self.calls = []

    def dispatch(self, name, arguments, state):
        self.calls.append((name, arguments))
        return {
            "ok": True,
            "tool_call": name,
            "target": arguments["target"],
            "summary": "fetched page",
            "result": {"title": "Demo"},
        }

    def tool_message_content(self, result):
        return json.dumps(result)


class FakeCatalog:
    class FakeManifest:
        def prompt_manifest(self, phase):
            return [
                {
                    "phase": "validation",
                    "tool": "sqlmap",
                    "profile": "basic_get_param_check",
                    "purpose": "Validate injectable GET parameters.",
                }
            ]

    manifest = FakeManifest()

    def openai_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_recon_fetch_page",
                    "description": "Fetch page",
                    "parameters": {
                        "type": "object",
                        "properties": {"target": {"type": "string"}},
                        "required": ["target"],
                    },
                },
            }
        ]


class FakeBadJsonThenValidLLM:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def chat_with_tools(self, messages, tools, max_tokens=1024, tool_choice="auto"):
        self.calls.append([dict(message) for message in messages])
        if len(self.calls) == 1:
            return {
                "role": "assistant",
                "content": '{"attack_surfaces": [], "test_plans": [}',
                "tool_calls": [],
                "finish_reason": "stop",
            }
        return {
            "role": "assistant",
            "content": json.dumps({"attack_surfaces": [], "test_plans": []}),
            "tool_calls": [],
            "finish_reason": "stop",
        }


class FakeMissingFieldThenValidLLM:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def chat_with_tools(self, messages, tools, max_tokens=1024, tool_choice="auto"):
        self.calls.append([dict(message) for message in messages])
        if len(self.calls) == 1:
            return {
                "role": "assistant",
                "content": json.dumps({"attack_surfaces": []}),
                "tool_calls": [],
                "finish_reason": "stop",
            }
        return {
            "role": "assistant",
            "content": json.dumps({"attack_surfaces": [], "test_plans": []}),
            "tool_calls": [],
            "finish_reason": "stop",
        }


class AgentToolLoopTests(unittest.TestCase):
    def test_tool_result_is_returned_to_llm_context(self) -> None:
        llm = FakeToolLLM()
        dispatcher = FakeDispatcher()
        loop = AgentToolLoop(llm_client=llm, catalog=FakeCatalog(), dispatcher=dispatcher)

        result = loop.run(
            system_prompt="Return JSON.",
            user_payload={"task": "inspect"},
            state={},
            final_repair_instruction="Return final JSON.",
        )

        self.assertEqual(dispatcher.calls[0][0], "web_recon_fetch_page")
        self.assertEqual(result.final, {"attack_surfaces": [], "test_plans": []})
        self.assertEqual(llm.calls[1][-1]["role"], "tool")
        self.assertEqual(llm.calls[1][-1]["tool_call_id"], "call_1")
        self.assertIn("fetched page", llm.calls[1][-1]["content"])

        first_payload = json.loads(llm.calls[0][1]["content"])
        self.assertIn("available_tool_manifest", first_payload)
        self.assertEqual(first_payload["available_tool_manifest"][0]["tool"], "sqlmap")
        self.assertFalse(first_payload["tool_execution_boundary"]["host_shell_available_to_llm"])

    def test_bad_json_repair_prompt_includes_error_and_schema_contract(self) -> None:
        llm = FakeBadJsonThenValidLLM()
        loop = AgentToolLoop(llm_client=llm, catalog=FakeCatalog(), dispatcher=FakeDispatcher())

        result = loop.run(
            system_prompt="Return JSON.",
            user_payload={
                "task": "inspect",
                "required_final_fields": ["attack_surfaces", "test_plans"],
                "final_output_schema": {"attack_surfaces": [], "test_plans": []},
            },
            state={},
            final_repair_instruction="Return discovery JSON.",
        )

        self.assertEqual(result.final, {"attack_surfaces": [], "test_plans": []})
        repair_prompt = llm.calls[1][-1]["content"]
        self.assertIn("Parser error", repair_prompt)
        self.assertIn("Required final JSON contract", repair_prompt)
        self.assertIn("attack_surfaces", repair_prompt)
        self.assertIn("test_plans", repair_prompt)
        self.assertIn("Return exactly one valid JSON object", repair_prompt)

    def test_missing_required_field_triggers_schema_repair(self) -> None:
        llm = FakeMissingFieldThenValidLLM()
        loop = AgentToolLoop(llm_client=llm, catalog=FakeCatalog(), dispatcher=FakeDispatcher())

        result = loop.run(
            system_prompt="Return JSON.",
            user_payload={
                "task": "inspect",
                "required_final_fields": ["attack_surfaces", "test_plans"],
                "json_contract": {
                    "required_top_level_fields": ["attack_surfaces", "test_plans"],
                    "arrays_may_be_empty": True,
                },
                "final_output_schema": {"attack_surfaces": [], "test_plans": []},
            },
            state={},
            final_repair_instruction="Return discovery JSON.",
        )

        self.assertEqual(result.final, {"attack_surfaces": [], "test_plans": []})
        repair_prompt = llm.calls[1][-1]["content"]
        self.assertIn("valid JSON, but it did not satisfy", repair_prompt)
        self.assertIn("Missing required top-level fields: test_plans", repair_prompt)
        self.assertIn("Keep arrays empty", repair_prompt)


if __name__ == "__main__":
    unittest.main()
