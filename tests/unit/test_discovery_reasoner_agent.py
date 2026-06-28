from __future__ import annotations

import json
import unittest

from autoflow.agents.discovery_reasoner import DiscoveryReasonerAgent
from autoflow.agents.tool_loop import ToolLoopResult
from autoflow.tools.catalog import ToolCatalog
from autoflow.tools.manifest import ToolManifestRegistry


class FakeConversationLLM:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def complete_messages(self, messages: list[dict[str, str]], max_tokens: int = 512) -> str:
        self.calls.append([dict(message) for message in messages])
        if len(self.calls) == 1:
            return json.dumps(
                {
                    "attack_surfaces": [
                        {
                            "target": "http://192.168.56.10:80",
                            "surface_type": "web_application",
                            "technology": "nginx",
                            "entrypoints": ["http://192.168.56.10:80"],
                            "related_assets": ["192.168.56.10:80"],
                            "rationale": "HTTP service and page title were observed.",
                        }
                    ]
                }
            )
        return json.dumps(
            {
                "test_plans": [
                    {
                        "target": "http://192.168.56.10:80",
                        "strategy": "llm_discovery",
                        "angle": "Use nuclei discovery templates on the web app.",
                        "risk_level": "low",
                        "requires_approval": False,
                        "rationale": "The target is a web application.",
                        "actions": [
                            {
                                "name": "Run nuclei discovery",
                                "action_kind": "tool",
                                "tool": "nuclei",
                                "profile": "discovery_all_severity",
                                "target": "http://192.168.56.10:80",
                                "risk_level": "low",
                                "requires_approval": False,
                            }
                        ],
                    }
                ]
            }
        )


class RepairingConversationLLM(FakeConversationLLM):
    def __init__(self) -> None:
        super().__init__()
        self.responses = [
            "I should inspect the context first.",
            "```text\nstill not json\n```",
            json.dumps(
                {
                    "attack_surfaces": [
                        {
                            "target": "http://192.168.56.10:80",
                            "surface_type": "web_application",
                            "technology": "nginx",
                            "entrypoints": ["http://192.168.56.10:80"],
                            "related_assets": ["192.168.56.10:80"],
                            "rationale": "HTTP service and page title were observed.",
                        }
                    ]
                }
            ),
            json.dumps({"test_plans": []}),
        ]

    def complete_messages(self, messages: list[dict[str, str]], max_tokens: int = 512) -> str:
        self.calls.append([dict(message) for message in messages])
        return self.responses[len(self.calls) - 1]


class FakeDiscoveryToolLoop:
    def __init__(self) -> None:
        self.catalog = ToolCatalog()
        self.calls: list[dict] = []

    def run(self, **kwargs) -> ToolLoopResult:
        self.calls.append(kwargs)
        return ToolLoopResult(
            final={
                "attack_surfaces": [
                    {
                        "target": "http://192.168.56.10:80",
                        "surface_type": "web_application",
                        "technology": "nginx",
                        "entrypoints": ["http://192.168.56.10:80"],
                        "related_assets": ["192.168.56.10:80"],
                        "rationale": "HTTP service and compact recon were observed.",
                    }
                ],
                "test_plans": [
                    {
                        "target": "http://192.168.56.10:80",
                        "strategy": "web_structure_discovery",
                        "angle": "Run read-only discovery checks.",
                        "risk_level": "low",
                        "requires_approval": False,
                        "rationale": "The target is a web application.",
                        "actions": [
                            {
                                "name": "Run nuclei discovery",
                                "action_kind": "tool",
                                "tool": "nuclei",
                                "profile": "discovery_all_severity",
                                "target": "http://192.168.56.10:80",
                                "risk_level": "low",
                                "requires_approval": False,
                            }
                        ],
                    }
                ],
            },
            messages=[],
            tool_results=[],
            iterations=1,
        )


class FailingDiscoveryToolLoop(FakeDiscoveryToolLoop):
    def run(self, **kwargs) -> ToolLoopResult:
        self.calls.append(kwargs)
        raise TimeoutError("simulated discovery LLM timeout")


class DiscoveryReasonerAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_reasoner_uses_one_conversation_for_surface_and_testplan_steps(self) -> None:
        llm = FakeConversationLLM()
        state = await DiscoveryReasonerAgent(llm_client=llm, use_llm=True, use_tool_calling=False).run(
            {
                "target_scope": ["192.168.56.10"],
                "rules_of_engagement": {"authorized": True},
                "assets": [
                    {
                        "ip": "192.168.56.10",
                        "ports": [
                            {
                                "port": 80,
                                "protocol": "tcp",
                                "state": "open",
                                "service": "http",
                                "product": "nginx",
                            }
                        ],
                    }
                ],
                "web_recon": [{"target": "http://192.168.56.10:80/", "title": "Demo"}],
                "max_rounds": 3,
            }
        )

        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[0][0]["role"], "system")
        self.assertEqual(llm.calls[1][0]["role"], "system")
        self.assertEqual(llm.calls[1][1]["role"], "user")
        self.assertEqual(llm.calls[1][2]["role"], "assistant")
        self.assertIn("attack_surfaces", llm.calls[1][2]["content"])
        self.assertEqual(state["attack_surfaces"][0]["metadata"]["source"], "llm_discovery_reasoner")
        self.assertEqual(state["test_plans"][0]["actions"][0]["tool"], "nuclei")
        self.assertIn("agent_memory", state)

    async def test_reasoner_repairs_invalid_json_in_same_conversation(self) -> None:
        llm = RepairingConversationLLM()
        state = await DiscoveryReasonerAgent(
            llm_client=llm,
            use_llm=True,
            json_repair_attempts=3,
            use_tool_calling=False,
        ).run(
            {
                "target_scope": ["192.168.56.10"],
                "assets": [
                    {
                        "ip": "192.168.56.10",
                        "ports": [{"port": 80, "protocol": "tcp", "state": "open", "service": "http"}],
                    }
                ],
                "web_recon": [{"target": "http://192.168.56.10:80/", "title": "Demo"}],
            }
        )

        self.assertGreaterEqual(len(llm.calls), 4)
        self.assertEqual(llm.calls[1][-2]["role"], "assistant")
        self.assertIn("I should inspect", llm.calls[1][-2]["content"])
        self.assertEqual(llm.calls[1][-1]["role"], "user")
        self.assertIn("could not be parsed", llm.calls[1][-1]["content"])
        self.assertEqual(llm.calls[2][-2]["role"], "assistant")
        self.assertIn("still not json", llm.calls[2][-2]["content"])
        self.assertEqual(state["attack_surfaces"][0]["target"], "http://192.168.56.10:80")

    async def test_reasoner_prompt_contains_tool_manifest_and_memory(self) -> None:
        llm = FakeConversationLLM()
        await DiscoveryReasonerAgent(llm_client=llm, use_llm=True, use_tool_calling=False).run(
            {
                "target_scope": ["192.168.56.10"],
                "assets": [],
                "web_recon": [],
                "tool_observations": [
                    {
                        "tool": "whatweb",
                        "profile": "web_fingerprint",
                        "target": "http://192.168.56.10",
                        "status": "completed",
                        "signals": [{"kind": "web_title", "name": "Demo"}],
                    }
                ],
            }
        )

        first_user_payload = json.loads(llm.calls[0][1]["content"])
        self.assertIn("memory", first_user_payload)
        self.assertIn("tool_manifest", first_user_payload)
        tools = {(item["tool"], item["profile"]) for item in first_user_payload["tool_manifest"]}
        self.assertIn(("nuclei", "discovery_all_severity"), tools)

    async def test_tool_loop_uses_compact_payload_and_discovery_only_tools(self) -> None:
        loop = FakeDiscoveryToolLoop()
        state = await DiscoveryReasonerAgent(use_llm=True, tool_loop=loop).run(
            {
                "target_scope": ["192.168.56.10"],
                "assets": [
                    {
                        "ip": "192.168.56.10",
                        "ports": [{"port": 80, "protocol": "tcp", "state": "open", "service": "http"}],
                    }
                ],
                "web_recon": [
                    {
                        "target": "http://192.168.56.10:80/",
                        "title": "Demo",
                        "links": [f"http://192.168.56.10:80/path-{index}" for index in range(120)],
                        "forms": [{"action": "http://192.168.56.10:80/login", "method": "post", "inputs": [1, 2]}],
                        "scripts": [f"/static/app-{index}.js" for index in range(40)],
                        "interesting_paths": [f"http://192.168.56.10:80/admin-{index}" for index in range(80)],
                    }
                ],
                "tool_observations": [
                    {
                        "tool": "nuclei",
                        "profile": "discovery_all_severity",
                        "target": "http://192.168.56.10:80",
                        "status": "completed",
                        "summary": "x" * 2000,
                        "signals": [
                            {"kind": "api_exposure", "name": f"api-{index}", "evidence": "y" * 1000}
                            for index in range(20)
                        ],
                    }
                ],
            }
        )

        call = loop.calls[0]
        payload = call["user_payload"]
        tool_names = {tool["function"]["name"] for tool in call["tools"]}
        self.assertIn("available_tool_manifest", payload)
        self.assertLessEqual(len(payload["web_recon"][0]["links"]), 40)
        self.assertLessEqual(len(payload["web_recon"][0]["scripts"]), 20)
        self.assertLessEqual(len(payload["tool_observations"][0]["signals"]), 12)
        self.assertEqual(payload["required_final_fields"], ["attack_surfaces", "test_plans"])
        self.assertEqual(payload["json_contract"]["required_top_level_fields"], ["attack_surfaces", "test_plans"])
        self.assertIn("run_nuclei__discovery_all_severity", tool_names)
        self.assertIn("web_recon_fetch_page", tool_names)
        self.assertNotIn("run_hydra__single_credential_check", tool_names)
        self.assertEqual(state["test_plans"][0]["actions"][0]["tool"], "nuclei")

    async def test_tool_loop_failure_falls_back_to_rule_based_discovery(self) -> None:
        loop = FailingDiscoveryToolLoop()
        state = await DiscoveryReasonerAgent(use_llm=True, tool_loop=loop).run(
            {
                "target_scope": ["192.168.56.10"],
                "assets": [
                    {
                        "ip": "192.168.56.10",
                        "ports": [{"port": 80, "protocol": "tcp", "state": "open", "service": "http"}],
                    }
                ],
                "web_recon": [{"target": "http://192.168.56.10:80/", "title": "Demo"}],
                "max_rounds": 3,
            }
        )

        self.assertEqual(state["discovery_reasoner_errors"][0]["fallback"], "rule_based_discovery")
        self.assertEqual(state["attack_surfaces"][0]["target"], "http://192.168.56.10:80")
        self.assertTrue(state["test_plans"])
        self.assertEqual(state["next_action"], "execute")

    def test_tool_manifest_registry_filters_discovery_tools(self) -> None:
        registry = ToolManifestRegistry()
        profiles = registry.allowed_profiles("discovery")

        self.assertIn(("tool", "nuclei", "discovery_all_severity"), profiles)
        self.assertIn(("web_recon", "web_recon", "fetch_page"), profiles)


if __name__ == "__main__":
    unittest.main()
