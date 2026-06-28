from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urljoin

from autoflow.agents.base import BaseAgent
from autoflow.agents.tool_loop import AgentToolLoop
from autoflow.flows.models import (
    AttackSurface,
    MemoryItem,
    MemoryKind,
    RiskLevel,
    TestPlan,
    TestPlanAction,
)
from autoflow.graph.state import AutoFlowState
from autoflow.llm.client import LLMClient
from autoflow.llm.client import parse_json_object
from autoflow.memory.agent_memory import AgentMemoryBuilder
from autoflow.memory.context import MemoryContextBuilder
from autoflow.runtime.actions import action_fingerprint, canonical_target
from autoflow.settings import settings
from autoflow.tools.manifest import ToolManifestRegistry


# AutoFlow 整体 pipeline 流转图：
#
#   User Target / Prompt
#          |
#          v
#   PlannerAgent
#     - 理解用户目标、范围、规则
#     - 创建 AssessmentFlow 和初始 recon 任务
#          |
#          v
#   DiscoveryAgent
#     |-- ReconAgent
#     |     - nmap / web_recon
#     |     - 产出 assets、web_recon、初始页面结构
#     |
#     `-- DiscoveryReasonerAgent  <-- 当前文件
#           - 读取 assets / web_recon / tool_observations / findings / memory
#           - 通过 LLM + tool calling 分析攻击面
#           - 生成 attack_surfaces 和 discovery TestPlan
#          |
#          v
#   ExecutorAgent
#     - 执行低风险 tool / script / shell / web_recon action
#     - medium/high action 进入 approvals_required
#     - 写入 executed_tasks、artifacts、tool_observations
#          |
#          v
#   VerifierAgent
#     - ToolObservation -> ToolSignal -> Candidate Finding
#          |
#          v
#   ValidationAgent
#     - Candidate Finding -> ValidationPlan
#     - 生成复现动作、成功/失败判据
#          |
#          v
#   ValidationExecutorAgent
#     - 执行验证动作
#     - 生成 ValidationResult
#     - 更新 Finding 状态：validated / false_positive / inconclusive
#          |
#          v
#   DiscoveryReasonerAgent / Strategy Loop
#     - 如果还有新 target/path/TestPlan，继续下一轮
#     - 如果没有新动作或达到 max_rounds，进入报告
#          |
#          v
#   ReporterAgent
#     - 汇总 assets、web_recon、findings、validation results、artifacts
#     - 输出 Markdown 报告
#
# 运行时记忆：
#   每个阶段的 state 会沉淀到 AgentMemoryBuilder / RedisMemoryStore。
#   LangGraph checkpointer 使用 thread_id 支持后续 resume。


WEB_LIKE_SERVICES = {"http", "https", "http-alt", "nessus"}
WEB_LIKE_PORTS = {80, 443, 3000, 3001, 5000, 8000, 8080, 8443, 8834}
DISCOVERY_TOOL_PHASES = {"discovery"}


DISCOVERY_REASONER_SYSTEM_PROMPT = """You are AutoFlow's DiscoveryReasonerAgent for an authorized security assessment.
Return only JSON. Do not include markdown.
You operate in a multi-turn conversation with the same context.
First analyze attack surfaces from recon and memory. Then generate discovery-stage TestPlans from that analysis.
Only use targets and paths present in authorized recon data or memory.
Only use the allowed tool profiles provided by the system.
Discovery actions must be read-only and low risk.
Do not invent external targets.
Do not create exploit, brute force, privilege escalation, lateral movement, persistence, destructive write, or evasion actions.
Candidate vulnerability validation is handled later by ValidationAgent.
"""


class DiscoveryReasonerAgent(BaseAgent):
    """单一 Discovery 推理 Agent：分析攻击面并生成发现阶段 TestPlan。"""

    name = "discovery_reasoner"

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        use_llm: bool | None = None,
        tool_manifest: ToolManifestRegistry | None = None,
        memory_builder: AgentMemoryBuilder | None = None,
        json_repair_attempts: int = 3,
        tool_loop: AgentToolLoop | None = None,
        use_tool_calling: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.tool_manifest = tool_manifest or ToolManifestRegistry()
        self.memory_builder = memory_builder or AgentMemoryBuilder()
        self.context_builder = MemoryContextBuilder()
        self.json_repair_attempts = json_repair_attempts
        self.tool_loop = tool_loop
        self.use_tool_calling = use_tool_calling

    async def run(self, state: AutoFlowState) -> AutoFlowState:
        state["current_phase"] = "discovery_reasoning"
        assets = state.get("assets", [])
        web_recon = state.get("web_recon", [])
        findings = state.get("findings", [])

        if self._should_use_llm():
            try:
                if self.use_tool_calling:
                    attack_surfaces, test_plans = self._reason_with_tool_loop(state, assets, web_recon, findings)
                else:
                    attack_surfaces, test_plans = self._reason_with_llm(state, assets, web_recon, findings)
            except Exception as exc:
                self._record_reasoning_error(state, exc)
                attack_surfaces = self._analyze_attack_surfaces_by_rules(assets, web_recon)
                test_plans = self._generate_test_plans_by_rules(state, attack_surfaces, web_recon, findings)
            state["attack_surfaces"] = [surface.model_dump(mode="json") for surface in attack_surfaces]
            state["agent_memory"] = self.memory_builder.build(state, persisted_memory=state.get("agent_memory"))
            state["memory_context"] = self.context_builder.build(state, persisted_memory=state["agent_memory"])
        else:
            attack_surfaces = self._analyze_attack_surfaces_by_rules(assets, web_recon)
            state["attack_surfaces"] = [surface.model_dump(mode="json") for surface in attack_surfaces]
            state["agent_memory"] = self.memory_builder.build(state, persisted_memory=state.get("agent_memory"))
            state["memory_context"] = self.context_builder.build(state, persisted_memory=state["agent_memory"])
            test_plans = self._generate_test_plans_by_rules(state, attack_surfaces, web_recon, findings)

        test_plans = self._dedupe_plans(test_plans)
        state["test_plans"] = [plan.model_dump(mode="json") for plan in test_plans]
        state["approvals_required"] = self._action_approval_requests(state.get("approvals_required", []))
        state["follow_up_tasks"] = []

        flow = state.get("flow")
        if flow is not None:
            for surface in attack_surfaces:
                flow.add_attack_surface(surface)
            for plan in test_plans:
                flow.add_test_plan(plan)
            flow.add_memory(
                MemoryItem(
                    kind=MemoryKind.DECISION,
                    content=(
                        f"DiscoveryReasoner identified {len(attack_surfaces)} attack surfaces "
                        f"and produced {len(test_plans)} test plans"
                    ),
                    source=self.name,
                    references=[*[surface.id for surface in attack_surfaces], *[plan.id for plan in test_plans]],
                )
            )

        strategy_round = int(state.get("strategy_round", 0)) + 1
        max_rounds = int(state.get("max_rounds", 3))
        state["strategy_round"] = strategy_round
        if strategy_round >= max_rounds:
            state["next_action"] = "report"
        else:
            state["next_action"] = "execute" if self._has_new_auto_actions(state, test_plans) else "report"
        state["current_phase"] = "discovery_reasoning"
        return state

    def _should_use_llm(self) -> bool:
        if self.use_llm is not None:
            return self.use_llm
        if not settings.llm_api_key:
            raise ValueError("LLM_API_KEY is required for DiscoveryReasonerAgent. Set use_llm=False for offline tests.")
        return True

    def _reason_with_tool_loop(
        self,
        state: AutoFlowState,
        assets: list[dict],
        web_recon: list[dict],
        findings: list[dict],
    ) -> tuple[list[AttackSurface], list[TestPlan]]:
        memory = self.memory_builder.build(state, persisted_memory=state.get("agent_memory"))
        loop = self.tool_loop or AgentToolLoop(
            llm_client=self.llm_client,
            max_tool_rounds=3,
            max_tool_calls=5,
            max_tokens=1024,
        )
        payload = {
            "task": "Produce compact discovery reasoning. Call tools only when current recon is insufficient.",
            "memory": self._compact_memory(memory),
            "available_tool_manifest": self._discovery_tool_manifest(loop),
            "tool_execution_boundary": {
                "containerized": True,
                "container_image": "autoflow-kali-tools",
                "host_shell_available_to_llm": False,
                "tools_filtered_for": "discovery",
            },
            "assets": self._compact_assets(assets),
            "web_recon": self._compact_web_recon(web_recon),
            "findings": self._compact_findings(findings),
            "tool_observations": self._compact_tool_observations(state.get("tool_observations", [])),
            "known_targets": sorted(self._allowed_targets_from_state(state)),
            "rules": {
                "authorized_assessment": True,
                "lab_mode": True,
                "use_tools_before_final_when_helpful": True,
                "targets_must_remain_authorized_or_discovered": True,
                "final_response_must_be_json": True,
            },
            "required_final_fields": ["attack_surfaces", "test_plans"],
            "json_contract": {
                "required_top_level_fields": ["attack_surfaces", "test_plans"],
                "return_only_json_object": True,
                "no_markdown": True,
                "arrays_may_be_empty": True,
            },
            "final_output_schema": {
                "attack_surfaces": [
                    {
                        "target": "authorized URL or host:port from recon/tool results",
                        "surface_type": "web_application",
                        "technology": "observed technology or empty string",
                        "entrypoints": ["authorized URL or service endpoint"],
                        "related_assets": ["host:port"],
                        "rationale": "why this is an attack surface based on evidence",
                        "metadata": {"source": "tool_calling_discovery_reasoner"},
                    }
                ],
                "test_plans": [
                    {
                        "target": "authorized target from context",
                        "strategy": "web_structure_discovery",
                        "angle": "why this step is useful",
                        "risk_level": "low",
                        "requires_approval": False,
                        "rationale": "short reason grounded in tool results",
                        "actions": [
                            {
                                "name": "Run discovery template checks",
                                "action_kind": "tool",
                                "tool": "nuclei",
                                "profile": "discovery_all_severity",
                                "target": "authorized target",
                                "risk_level": "low",
                                "requires_approval": False,
                                "args": {},
                                "script_template": None,
                                "rationale": "short reason",
                            }
                        ],
                    }
                ],
            },
        }
        result = loop.run(
            system_prompt=DISCOVERY_REASONER_SYSTEM_PROMPT,
            user_payload=payload,
            state=state,
            final_repair_instruction=(
                "Return a JSON object with attack_surfaces and test_plans arrays. "
                "Do not include exploit, brute force, persistence, destructive, or out-of-scope actions."
            ),
            tools=loop.catalog.openai_tools(DISCOVERY_TOOL_PHASES),
        )
        state["tool_loop_messages"] = result.messages
        state["tool_loop_results"] = result.tool_results

        surface_items = result.final.get("attack_surfaces", [])
        if not isinstance(surface_items, list):
            surface_items = []
        attack_surfaces = self._coerce_surfaces(surface_items, state.get("assets", []), state.get("web_recon", []))
        plan_items = result.final.get("test_plans", [])
        if not isinstance(plan_items, list):
            plan_items = []
        return attack_surfaces, self._coerce_plans(plan_items, state, attack_surfaces)

    def _record_reasoning_error(self, state: AutoFlowState, exc: Exception) -> None:
        errors = list(state.get("discovery_reasoner_errors", []))
        errors.append(
            {
                "phase": "discovery_reasoning",
                "error": str(exc),
                "fallback": "rule_based_discovery",
            }
        )
        state["discovery_reasoner_errors"] = errors[-10:]

    def _discovery_tool_manifest(self, loop: AgentToolLoop) -> list[dict[str, Any]]:
        manifest: list[dict[str, Any]] = []
        for function in loop.catalog.functions(DISCOVERY_TOOL_PHASES):
            metadata = function.metadata
            manifest.append(
                {
                    "function": function.name,
                    "tool": metadata.get("tool"),
                    "profile": metadata.get("profile"),
                    "kind": metadata.get("kind"),
                    "risk_level": metadata.get("risk_level"),
                    "purpose": self._truncate(function.description, 420),
                }
            )
        return manifest

    def _compact_memory(self, memory: dict[str, Any]) -> dict[str, Any]:
        return {
            "target_scope": memory.get("target_scope", [])[:20],
            "assets": self._compact_assets(memory.get("assets", [])),
            "web_context": self._compact_web_recon(memory.get("web_context", [])),
            "attack_surfaces": [
                {
                    "target": item.get("target"),
                    "surface_type": item.get("surface_type"),
                    "technology": item.get("technology", ""),
                    "entrypoints": item.get("entrypoints", [])[:20],
                }
                for item in memory.get("attack_surfaces", [])[-20:]
                if isinstance(item, dict)
            ],
            "findings": self._compact_findings(memory.get("findings", [])),
            "validation_results": [
                {
                    "finding_id": item.get("finding_id"),
                    "status": item.get("status"),
                    "confidence": item.get("confidence"),
                    "reasoning": self._truncate(item.get("reasoning", ""), 500),
                }
                for item in memory.get("validation_results", [])[-20:]
                if isinstance(item, dict)
            ],
            "tool_observations": self._compact_tool_observations(memory.get("tool_observations", [])),
        }

    def _compact_assets(self, assets: list[dict]) -> list[dict]:
        compact = []
        for asset in assets[-30:]:
            if not isinstance(asset, dict):
                continue
            compact.append(
                {
                    "ip": asset.get("ip"),
                    "hostnames": asset.get("hostnames", [])[:10],
                    "ports": [
                        {
                            "port": port.get("port"),
                            "protocol": port.get("protocol"),
                            "state": port.get("state"),
                            "service": port.get("service"),
                            "product": port.get("product", ""),
                            "version": port.get("version", ""),
                        }
                        for port in asset.get("ports", [])[:30]
                        if isinstance(port, dict)
                    ],
                }
            )
        return compact

    def _compact_web_recon(self, web_recon: list[dict]) -> list[dict]:
        compact = []
        for item in web_recon[-30:]:
            if not isinstance(item, dict):
                continue
            robots = item.get("robots") if isinstance(item.get("robots"), dict) else {}
            compact.append(
                {
                    "target": canonical_target(str(item.get("target", ""))),
                    "status_code": item.get("status_code"),
                    "title": item.get("title"),
                    "links": [canonical_target(str(value)) for value in item.get("links", [])[:40]],
                    "forms": [
                        {
                            "action": canonical_target(str(form.get("action", ""))),
                            "method": form.get("method", ""),
                            "inputs": len(form.get("inputs", [])),
                        }
                        for form in item.get("forms", [])[:10]
                        if isinstance(form, dict)
                    ],
                    "scripts": item.get("scripts", [])[:20],
                    "interesting_paths": [
                        canonical_target(str(value))
                        for value in item.get("interesting_paths", [])[:40]
                    ],
                    "robots_interesting_paths": [
                        canonical_target(str(value))
                        for value in robots.get("interesting_paths", [])[:40]
                    ],
                    "error": item.get("error", ""),
                }
            )
        return compact

    def _compact_findings(self, findings: list[dict]) -> list[dict]:
        compact = []
        for item in findings[-40:]:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            compact.append(
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "status": item.get("status"),
                    "severity": item.get("severity"),
                    "confidence": item.get("confidence"),
                    "target": canonical_target(str(item.get("target", ""))),
                    "category": metadata.get("category"),
                    "evidence": [self._truncate(str(value), 500) for value in item.get("evidence", [])[:5]],
                    "source": item.get("source", ""),
                }
            )
        return compact

    def _compact_tool_observations(self, observations: list[dict]) -> list[dict]:
        compact = []
        for item in observations[-40:]:
            if not isinstance(item, dict):
                continue
            compact.append(
                {
                    "tool": item.get("tool"),
                    "profile": item.get("profile"),
                    "target": canonical_target(str(item.get("target", ""))),
                    "status": item.get("status"),
                    "summary": self._truncate(item.get("summary", ""), 500),
                    "signals": [
                        {
                            "kind": signal.get("kind"),
                            "name": signal.get("name"),
                            "severity": signal.get("severity"),
                            "target": canonical_target(str(signal.get("target", ""))),
                            "evidence": self._truncate(signal.get("evidence", ""), 400),
                        }
                        for signal in item.get("signals", [])[:12]
                        if isinstance(signal, dict)
                    ],
                }
            )
        return compact

    def _truncate(self, value: object, limit: int) -> str:
        text = str(value or "")
        if len(text) <= limit:
            return text
        return text[: limit - 15] + "...[truncated]"

    def _reason_with_llm(
        self,
        state: AutoFlowState,
        assets: list[dict],
        web_recon: list[dict],
        findings: list[dict],
    ) -> tuple[list[AttackSurface], list[TestPlan]]:
        client = self.llm_client or LLMClient()
        memory = self.memory_builder.build(state, persisted_memory=state.get("agent_memory"))
        tool_manifest = self.tool_manifest.prompt_manifest("discovery")
        messages = [{"role": "system", "content": DISCOVERY_REASONER_SYSTEM_PROMPT}]

        surface_prompt = {
            "step": "attack_surface_analysis",
            "memory": self._compact_memory(memory),
            "tool_manifest": tool_manifest,
            "assets": self._compact_assets(assets),
            "web_recon": self._compact_web_recon(web_recon),
            "allowed_surface_types": ["web_application", "network_service", "api", "static_content", "unknown"],
            "output_schema": {
                "attack_surfaces": [
                    {
                        "target": "authorized URL or host:port from recon data",
                        "surface_type": "web_application",
                        "technology": "observed technology or empty string",
                        "entrypoints": ["authorized URL or service endpoint"],
                        "related_assets": ["host:port"],
                        "rationale": "why this is an attack surface based on recon evidence",
                        "metadata": {"source": "llm_discovery_reasoner"},
                    }
                ]
            },
        }
        messages.append({"role": "user", "content": json.dumps(surface_prompt, ensure_ascii=False)})
        surface_content, surface_response = self._complete_json_in_context(
            client=client,
            messages=messages,
            max_tokens=1024,
            repair_instruction="Return only a valid JSON object with an attack_surfaces array.",
        )
        messages.append({"role": "assistant", "content": surface_content})
        items = surface_response.get("attack_surfaces", [])
        if not isinstance(items, list):
            items = []
        attack_surfaces = self._coerce_surfaces(items, assets, web_recon)

        testplan_prompt = {
            "step": "test_plan_generation",
            "memory": {
                **self._compact_memory(memory),
                "attack_surfaces": [surface.model_dump(mode="json") for surface in attack_surfaces],
            },
            "tool_manifest": tool_manifest,
            "attack_surfaces": [surface.model_dump(mode="json") for surface in attack_surfaces],
            "web_recon": self._compact_web_recon(web_recon),
            "findings": self._compact_findings(findings),
            "tool_observations": self._compact_tool_observations(state.get("tool_observations", [])),
            "output_schema": {
                "test_plans": [
                    {
                        "target": "authorized target from context",
                        "strategy": "web_structure_discovery",
                        "angle": "why this discovery step is useful",
                        "risk_level": "low",
                        "requires_approval": False,
                        "rationale": "short reason",
                        "actions": [
                            {
                                "name": "Run discovery template checks",
                                "action_kind": "tool",
                                "tool": "nuclei",
                                "profile": "discovery_all_severity",
                                "target": "authorized target from context",
                                "risk_level": "low",
                                "requires_approval": False,
                                "args": {},
                                "script_template": None,
                                "rationale": "short reason",
                            }
                        ],
                    }
                ]
            },
        }
        messages.append({"role": "user", "content": json.dumps(testplan_prompt, ensure_ascii=False)})
        _testplan_content, testplan_response = self._complete_json_in_context(
            client=client,
            messages=messages,
            max_tokens=1024,
            repair_instruction="Return only a valid JSON object with a test_plans array.",
        )
        items = testplan_response.get("test_plans", [])
        if not isinstance(items, list):
            items = []
        return attack_surfaces, self._coerce_plans(items, state, attack_surfaces)

    def _complete_json_in_context(
        self,
        *,
        client: LLMClient,
        messages: list[dict[str, str]],
        max_tokens: int,
        repair_instruction: str,
    ) -> tuple[str, dict[str, Any]]:
        last_error: Exception | None = None
        max_attempts = max(1, self.json_repair_attempts + 1)
        for attempt in range(max_attempts):
            content = client.complete_messages(messages=messages, max_tokens=max_tokens)
            try:
                return content, parse_json_object(content)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt >= max_attempts - 1:
                    break
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response could not be parsed as a JSON object. "
                            f"Parser error: {exc}. {repair_instruction} "
                            "Return exactly one JSON object. Do not include markdown fences, comments, "
                            "analysis text, apologies, or explanations."
                        ),
                    }
                )
        raise ValueError(
            f"DiscoveryReasonerAgent failed to obtain valid JSON after {max_attempts} attempts: {last_error}"
        )

    def _coerce_surfaces(
        self,
        items: list[dict[str, Any]],
        assets: list[dict],
        web_recon: list[dict],
    ) -> list[AttackSurface]:
        allowed_targets = self._allowed_targets_from_recon(assets, web_recon)
        surfaces: list[AttackSurface] = []
        for item in items:
            target = canonical_target(str(item.get("target", "")))
            if not target or target not in allowed_targets:
                continue
            entrypoints = [
                canonical_target(str(value))
                for value in item.get("entrypoints", [])
                if canonical_target(str(value)) in allowed_targets
            ]
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            surfaces.append(
                AttackSurface(
                    target=target,
                    surface_type=str(item.get("surface_type") or "unknown"),
                    technology=str(item.get("technology") or ""),
                    entrypoints=entrypoints or [target],
                    related_assets=[str(value) for value in item.get("related_assets", [])][:20],
                    rationale=str(item.get("rationale") or "LLM identified this surface from recon evidence."),
                    metadata={**metadata, "source": "llm_discovery_reasoner"},
                )
            )
        return surfaces

    def _coerce_plans(
        self,
        items: list[dict[str, Any]],
        state: AutoFlowState,
        attack_surfaces: list[AttackSurface] | None = None,
    ) -> list[TestPlan]:
        allowed_targets = self._allowed_targets_from_state(state, attack_surfaces or [])
        plans: list[TestPlan] = []
        for item in items:
            target = canonical_target(str(item.get("target", "")))
            if not target or target not in allowed_targets:
                continue
            actions = self._coerce_actions(item.get("actions", []), allowed_targets)
            if not actions:
                continue
            plans.append(
                TestPlan(
                    target=target,
                    strategy=str(item.get("strategy") or "llm_discovery"),
                    angle=str(item.get("angle") or "LLM selected a safe discovery angle"),
                    risk_level=RiskLevel.LOW,
                    requires_approval=False,
                    actions=actions,
                    rationale=str(item.get("rationale") or "Generated by LLM discovery reasoner."),
                    metadata={"source": "llm_discovery_reasoner"},
                )
            )
        return plans

    def _coerce_actions(self, items: Any, allowed_targets: set[str]) -> list[TestPlanAction]:
        if not isinstance(items, list):
            return []
        actions: list[TestPlanAction] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            action_kind = str(item.get("action_kind") or "tool")
            tool = str(item.get("tool") or "")
            profile = str(item.get("profile") or "")
            target = canonical_target(str(item.get("target") or ""))
            if (action_kind, tool, profile) not in self.tool_manifest.allowed_profiles("discovery"):
                continue
            if not target or target not in allowed_targets:
                continue
            args = item.get("args") if isinstance(item.get("args"), dict) else {}
            if tool == "nikto":
                args = {"maxtime": str(args.get("maxtime", "60"))}
            script_template = None
            if tool == "script_runner":
                script_template = item.get("script_template") or (
                    "security_headers_check" if profile == "security_headers_check" else None
                )
            actions.append(
                TestPlanAction(
                    name=str(item.get("name") or f"Run {tool} {profile}"),
                    action_kind=action_kind,
                    tool=tool,
                    profile=profile,
                    target=target,
                    risk_level=RiskLevel.LOW,
                    requires_approval=False,
                    expected_impact=str(item.get("expected_impact") or "Read-only discovery action."),
                    rationale=str(item.get("rationale") or "LLM selected this read-only discovery action."),
                    args={str(key): str(value) for key, value in args.items()},
                    script_template=script_template,
                    metadata={"source": "llm_discovery_reasoner"},
                )
            )
        return actions

    def _analyze_attack_surfaces_by_rules(self, assets: list[dict], web_recon: list[dict]) -> list[AttackSurface]:
        surfaces: list[AttackSurface] = []
        recon_by_target = {canonical_target(str(item.get("target", ""))): item for item in web_recon or []}
        for asset in assets:
            host = asset.get("ip")
            if not host:
                continue
            for port in asset.get("ports", []):
                port_number = port.get("port")
                if not port_number:
                    continue
                port_number_int = int(port_number)
                service = (port.get("service") or "").lower()
                product = port.get("product") or ""
                target = f"{host}:{port_number_int}"
                if self._is_web_like(service, port_number_int):
                    scheme = "https" if service == "https" or port_number_int in {443, 8443} else "http"
                    entrypoint = f"{scheme}://{target}"
                    surfaces.append(
                        AttackSurface(
                            target=entrypoint,
                            surface_type="web_application",
                            technology=product or service,
                            entrypoints=self._web_entrypoints(entrypoint, recon_by_target.get(entrypoint)),
                            related_assets=[target],
                            rationale=f"{service or port_number_int} appears web-like and can be fingerprinted safely.",
                            metadata={"asset": asset, "port": port, "web_recon": recon_by_target.get(entrypoint)},
                        )
                    )
                else:
                    surfaces.append(
                        AttackSurface(
                            target=target,
                            surface_type="network_service",
                            technology=product or service,
                            entrypoints=[target],
                            related_assets=[target],
                            rationale="Open service should be reviewed before deeper validation.",
                            metadata={"asset": asset, "port": port},
                        )
                    )
        return surfaces

    def _generate_test_plans_by_rules(
        self,
        state: AutoFlowState,
        attack_surfaces: list[AttackSurface],
        web_recon: list[dict],
        findings: list[dict],
    ) -> list[TestPlan]:
        plans: list[TestPlan] = []
        for surface_model in attack_surfaces:
            surface = surface_model.model_dump(mode="json")
            if surface.get("surface_type") == "web_application":
                plans.append(self._web_fingerprint_plan(surface))
            elif surface.get("surface_type") == "network_service":
                plans.append(self._service_review_plan(surface))

        for item in web_recon:
            if self._web_recon_has_structure(item):
                plans.append(self._web_structure_discovery_plan(item))

        for target in self._discovered_path_targets(web_recon, findings):
            plans.append(self._web_recon_refresh_plan(target))
        return plans

    def _web_fingerprint_plan(self, surface: dict) -> TestPlan:
        target = surface.get("target", "unknown")
        return TestPlan(
            target=target,
            strategy="web_fingerprinting",
            angle="Identify web application technology and exposure clues",
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            actions=[
                TestPlanAction(
                    name="Run web fingerprinting",
                    action_kind="tool",
                    tool="whatweb",
                    profile="web_fingerprint",
                    target=target,
                    risk_level=RiskLevel.LOW,
                    requires_approval=False,
                    expected_impact="One or a few read-only HTTP requests.",
                    rationale="Fingerprint the web surface before selecting deeper checks.",
                )
            ],
            rationale="A web-like attack surface was found during discovery reasoning.",
            metadata={"attack_surface": surface, "source": "rule_fallback"},
        )

    def _service_review_plan(self, surface: dict) -> TestPlan:
        target = surface.get("target", "unknown")
        return TestPlan(
            target=target,
            strategy="service_exposure_review",
            angle="Exposed network service review",
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            actions=[],
            rationale="The service is noted for review; no additional automatic tool is selected yet.",
            metadata={"attack_surface": surface, "source": "rule_fallback"},
        )

    def _web_structure_discovery_plan(self, web_recon: dict) -> TestPlan:
        target = canonical_target(str(web_recon.get("target", "unknown")))
        nikto_maxtime = self._nikto_maxtime(web_recon)
        return TestPlan(
            target=target,
            strategy="web_structure_discovery",
            angle="Discover web structure, entrypoints, and common exposure clues",
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            actions=[
                TestPlanAction(
                    name="Run basic web configuration checks",
                    action_kind="tool",
                    tool="nikto",
                    profile="basic_web_check",
                    target=target,
                    risk_level=RiskLevel.LOW,
                    requires_approval=False,
                    expected_impact="Read-only HTTP requests against discovered web application.",
                    rationale="Web recon found a reachable application surface.",
                    args={"maxtime": str(nikto_maxtime)},
                    metadata={"web_recon": web_recon, "source": "rule_fallback"},
                ),
                TestPlanAction(
                    name="Run discovery template checks",
                    action_kind="tool",
                    tool="nuclei",
                    profile="discovery_all_severity",
                    target=target,
                    risk_level=RiskLevel.LOW,
                    requires_approval=False,
                    expected_impact="Read-only nuclei discovery templates across all severities.",
                    rationale="Collect candidate findings with safe read-only nuclei templates.",
                    metadata={"web_recon": web_recon, "source": "rule_fallback"},
                ),
                TestPlanAction(
                    name="Check security headers from discovered landing page",
                    action_kind="script",
                    tool="script_runner",
                    profile="security_headers_check",
                    target=target,
                    risk_level=RiskLevel.LOW,
                    requires_approval=False,
                    expected_impact="One read-only HTTP GET and header parsing.",
                    rationale="Security headers are useful discovery context before validation.",
                    script_template="security_headers_check",
                    metadata={"web_recon": web_recon, "source": "rule_fallback"},
                ),
            ],
            rationale=(
                f"Web recon observed title={web_recon.get('title')!r}, "
                f"{len(web_recon.get('links', []))} links, {len(web_recon.get('forms', []))} forms, "
                f"{len(web_recon.get('scripts', []))} scripts, and "
                f"{len(web_recon.get('interesting_paths', []))} interesting paths."
            ),
            metadata={"web_recon": web_recon, "source": "rule_fallback"},
        )

    def _web_recon_refresh_plan(self, target: str) -> TestPlan:
        return TestPlan(
            target=target,
            strategy="web_recon_refresh",
            angle="Fetch and parse a newly discovered web path",
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            actions=[
                TestPlanAction(
                    name="Refresh web recon for discovered path",
                    action_kind="web_recon",
                    tool="web_recon",
                    profile="fetch_page",
                    target=target,
                    risk_level=RiskLevel.LOW,
                    requires_approval=False,
                    expected_impact="One read-only HTTP GET plus robots/sitemap checks.",
                    rationale="A previous observation exposed this path or route.",
                )
            ],
            rationale="New web path or route discovered from observations/findings.",
        )

    def _allowed_targets_from_recon(self, assets: list[dict], web_recon: list[dict]) -> set[str]:
        targets: set[str] = set()
        for asset in assets:
            host = asset.get("ip")
            if not host:
                continue
            targets.add(str(host))
            for port in asset.get("ports", []):
                port_number = port.get("port")
                if not port_number:
                    continue
                port_number_int = int(port_number)
                service_target = f"{host}:{port_number_int}"
                targets.add(service_target)
                service = (port.get("service") or "").lower()
                if self._is_web_like(service, port_number_int):
                    scheme = "https" if service == "https" or port_number_int in {443, 8443} else "http"
                    targets.add(f"{scheme}://{service_target}")
        for item in web_recon:
            if item.get("target"):
                targets.add(canonical_target(str(item["target"])))
            for key in ("links", "interesting_paths"):
                for value in item.get(key, [])[:100]:
                    if value:
                        targets.add(canonical_target(str(value)))
            robots = item.get("robots") or {}
            for value in robots.get("interesting_paths", [])[:100]:
                targets.add(canonical_target(str(value)))
            for form in item.get("forms", [])[:50]:
                action = form.get("action")
                if action:
                    targets.add(canonical_target(str(action)))
        return {target for target in targets if target}

    def _allowed_targets_from_state(
        self,
        state: AutoFlowState,
        attack_surfaces: list[AttackSurface] | None = None,
    ) -> set[str]:
        targets = self._allowed_targets_from_recon(state.get("assets", []), state.get("web_recon", []))
        for surface in attack_surfaces or []:
            targets.add(canonical_target(surface.target))
            for entrypoint in surface.entrypoints:
                targets.add(canonical_target(entrypoint))
        for surface in state.get("attack_surfaces", []):
            if surface.get("target"):
                targets.add(canonical_target(str(surface["target"])))
            for entrypoint in surface.get("entrypoints", []):
                targets.add(canonical_target(str(entrypoint)))
        for finding in state.get("findings", []):
            if finding.get("target"):
                targets.add(canonical_target(str(finding["target"])))
        return {target for target in targets if target}

    def _discovered_path_targets(self, web_recon: list[dict], findings: list[dict]) -> list[str]:
        targets = set()
        for item in web_recon:
            robots = item.get("robots") or {}
            targets.update(robots.get("interesting_paths", []))
            targets.update(item.get("interesting_paths", []))

        for finding in findings:
            title = finding.get("title", "")
            if not title.startswith("Informational x-recruiting"):
                continue
            base = finding.get("target", "")
            for evidence in finding.get("evidence", []):
                for route in re.findall(r"(/[A-Za-z0-9_./#?=&%-]+)", evidence):
                    route = route.rstrip(".,;)")
                    if route in {"", "/"} or route.startswith("//"):
                        continue
                    targets.add(canonical_target(urljoin(base, route)))

        return [
            canonical_target(target)
            for target in sorted({canonical_target(str(target)) for target in targets})
            if canonical_target(str(target)).startswith(("http://", "https://"))
        ]

    def _web_entrypoints(self, entrypoint: str, web_recon: dict | None) -> list[str]:
        if not web_recon:
            return [entrypoint]
        values = [
            entrypoint,
            *web_recon.get("links", []),
            *[form.get("action", "") for form in web_recon.get("forms", [])],
            *web_recon.get("interesting_paths", []),
        ]
        return [canonical_target(str(value)) for value in dict.fromkeys(values) if value][:100]

    def _web_recon_has_structure(self, web_recon: dict) -> bool:
        return any(
            [
                web_recon.get("links"),
                web_recon.get("forms"),
                web_recon.get("scripts"),
                web_recon.get("interesting_paths"),
                web_recon.get("title"),
            ]
        )

    def _is_web_like(self, service: str, port_number: int) -> bool:
        return service in WEB_LIKE_SERVICES or port_number in WEB_LIKE_PORTS

    def _dedupe_plans(self, plans: list[TestPlan]) -> list[TestPlan]:
        seen: set[tuple[str, str]] = set()
        result: list[TestPlan] = []
        for plan in plans:
            key = (plan.strategy, canonical_target(plan.target))
            if key in seen:
                continue
            seen.add(key)
            result.append(plan)
        return result

    def _action_approval_requests(self, approvals: list[dict]) -> list[dict]:
        result: list[dict] = []
        seen: set[str] = set()
        for item in approvals:
            action_id = item.get("action_id")
            if not action_id or action_id in seen:
                continue
            seen.add(action_id)
            result.append(item)
        return result

    def _has_new_auto_actions(self, state: AutoFlowState, plans: list[TestPlan]) -> bool:
        executed = set(state.get("executed_action_fingerprints", []))
        for plan in plans:
            for action in plan.actions:
                if action.requires_approval or action.risk_level != RiskLevel.LOW:
                    continue
                if action_fingerprint(action.model_dump(mode="json")) not in executed:
                    return True
        return False

    def _nikto_maxtime(self, web_recon: dict) -> int:
        size = {
            "links": len(web_recon.get("links", [])),
            "forms": len(web_recon.get("forms", [])),
            "scripts": len(web_recon.get("scripts", [])),
            "interesting_paths": len(web_recon.get("interesting_paths", [])),
        }
        score = size["links"] + size["interesting_paths"] + size["scripts"] + size["forms"] * 3
        if score <= 10:
            return 60
        if score <= 50:
            return 90
        if score <= 150:
            return 120
        return 180
