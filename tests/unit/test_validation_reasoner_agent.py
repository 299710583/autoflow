from __future__ import annotations

import unittest

from autoflow.agents.tool_loop import ToolLoopResult
from autoflow.agents.validation_reasoner import ValidationReActReasoner
from autoflow.tools.catalog import ToolCatalog


class FakeValidationToolLoop:
    def __init__(self) -> None:
        self.catalog = ToolCatalog()
        self.calls = []

    def run(self, **kwargs) -> ToolLoopResult:
        self.calls.append(kwargs)
        return ToolLoopResult(
            final={
                "decision": "inconclusive",
                "confidence": "medium",
                "reasoning": "Need more evidence.",
                "impact": "",
                "evidence": [],
                "missing_evidence": ["No validation action has been run yet."],
                "reproduction_steps": [],
                "next_actions": [],
            },
            messages=[],
            tool_results=[],
            iterations=1,
        )


class ValidationReasonerAgentTests(unittest.TestCase):
    def test_payload_exposes_full_validation_toolbox_and_context(self) -> None:
        loop = FakeValidationToolLoop()
        reasoner = ValidationReActReasoner(tool_loop=loop)
        finding = {
            "id": "finding-1",
            "title": "Possible SQL injection in search endpoint",
            "target": "http://example.test/search?q=juice",
            "description": "Scanner observed SQL error hints.",
            "evidence": ["nuclei reported SQL error pattern on /search?q=juice"],
            "metadata": {"category": "sql_injection"},
        }

        reasoner.validate(
            state={
                "target_scope": ["http://example.test"],
                "findings": [finding],
                "web_recon": [
                    {
                        "target": "http://example.test",
                        "title": "Demo",
                        "links": ["http://example.test/search?q=juice"],
                        "forms": [],
                        "interesting_paths": ["http://example.test/rest/products/search?q=apple"],
                    }
                ],
                "tool_observations": [
                    {
                        "tool": "nuclei",
                        "profile": "discovery_all_severity",
                        "target": "http://example.test/search?q=juice",
                        "status": "completed",
                        "summary": "SQL error pattern observed.",
                        "signals": [
                            {
                                "kind": "sql_injection",
                                "name": "SQL error",
                                "severity": "high",
                                "target": "http://example.test/search?q=juice",
                                "evidence": "SQL syntax error",
                            }
                        ],
                    }
                ],
            },
            finding=finding,
        )

        call = loop.calls[0]
        payload = call["user_payload"]
        tool_names = {tool["function"]["name"] for tool in call["tools"]}
        manifest_names = {item["function"] for item in payload["available_tool_manifest"]}

        self.assertIn("run_sqlmap__basic_get_param_check", tool_names)
        self.assertIn("run_script__custom_validation", tool_names)
        self.assertIn("run_shell__bounded_bash", tool_names)
        self.assertIn("run_script__custom_validation", manifest_names)
        self.assertIn("run_sqlmap__basic_get_param_check", payload["recommended_tools"])
        self.assertIn("run_script__custom_validation", payload["recommended_tools"])
        self.assertEqual(payload["validation_context"]["target_host"], "example.test")
        self.assertTrue(payload["validation_context"]["related_observations"])
        self.assertIn("custom_script_contract", payload)


if __name__ == "__main__":
    unittest.main()
