from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urljoin

from autoflow.agents.base import BaseAgent
from autoflow.flows.models import MemoryItem, MemoryKind, RiskLevel, TestPlan, TestPlanAction
from autoflow.graph.state import AutoFlowState
from autoflow.llm.client import LLMClient
from autoflow.memory.context import MemoryContextBuilder
from autoflow.runtime.actions import action_fingerprint, canonical_target
from autoflow.settings import settings


STRATEGIST_SYSTEM_PROMPT = """You are AutoFlow's discovery strategy agent for an authorized security assessment.
Return only JSON. Do not include markdown.
Use recon, web_recon, attack surfaces, observations, and findings to generate discovery-stage TestPlans.
Only use the allowed tool profiles provided by the system.
Discovery actions must be read-only and low risk. Do not plan exploitation, brute force, privilege escalation, lateral movement, persistence, destructive writes, or evasion.
Candidate vulnerability validation is handled by ValidationAgent; do not create exploit or PoC actions here.
"""


ALLOWED_ACTION_PROFILES = {
    ("tool", "whatweb", "web_fingerprint"),
    ("tool", "nikto", "basic_web_check"),
    ("tool", "nuclei", "discovery_all_severity"),
    ("script", "script_runner", "security_headers_check"),
    ("web_recon", "web_recon", "fetch_page"),
}


class StrategistAgent(BaseAgent):
    """根据攻击面和 Findings 生成受控 TestPlan。"""

    name = "strategist"

    def __init__(self, use_llm: bool | None = None, llm_client: LLMClient | None = None) -> None:
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.context_builder = MemoryContextBuilder()

    async def run(self, state: AutoFlowState) -> AutoFlowState:
        state["current_phase"] = "strategy"
        state["memory_context"] = self.context_builder.build(state)
        strategy_round = int(state.get("strategy_round", 0)) + 1
        max_rounds = int(state.get("max_rounds", 3))
        state["strategy_round"] = strategy_round
        attack_surfaces = state.get("attack_surfaces", [])
        findings = state.get("findings", [])
        web_recon = state.get("web_recon", [])
        if self._should_use_llm():
            test_plans = self._plans_with_llm(state, attack_surfaces, web_recon, findings)
        else:
            test_plans = self._rule_based_plans(attack_surfaces, web_recon, findings)
        test_plans = self._dedupe_plans(test_plans)

        state["test_plans"] = [plan.model_dump(mode="json") for plan in test_plans]
        state["approvals_required"] = self._action_approval_requests(state.get("approvals_required", []))

        flow = state.get("flow")
        if flow is not None:
            for plan in test_plans:
                flow.add_test_plan(plan)
            flow.add_memory(
                MemoryItem(
                    kind=MemoryKind.DECISION,
                    content=f"Strategist produced {len(test_plans)} test plans",
                    source=self.name,
                    references=[plan.id for plan in test_plans],
                )
            )

        if strategy_round >= max_rounds:
            state["next_action"] = "report"
        else:
            state["next_action"] = "execute" if self._has_new_auto_actions(state, test_plans) else "report"
        return state

    def _should_use_llm(self) -> bool:
        if self.use_llm is not None:
            return self.use_llm
        if not settings.llm_api_key:
            raise ValueError("LLM_API_KEY is required for StrategistAgent. Set use_llm=False for offline tests.")
        return True

    def _rule_based_plans(
        self,
        attack_surfaces: list[dict],
        web_recon: list[dict],
        findings: list[dict],
    ) -> list[TestPlan]:
        return [
            *self._plans_from_attack_surfaces(attack_surfaces),
            *self._plans_from_web_recon(web_recon),
            *self._plans_from_discovered_paths(web_recon, findings),
            *self._plans_from_findings(findings),
        ]

    def _plans_with_llm(
        self,
        state: AutoFlowState,
        attack_surfaces: list[dict],
        web_recon: list[dict],
        findings: list[dict],
    ) -> list[TestPlan]:
        client = self.llm_client or LLMClient()
        prompt = {
            "user_prompt": state.get("user_prompt", ""),
            "rules_of_engagement": state.get("rules_of_engagement", {}),
            "memory_context": state.get("memory_context", {}),
            "attack_surfaces": attack_surfaces,
            "web_recon": web_recon,
            "findings": findings,
            "tool_observations": state.get("tool_observations", []),
            "already_executed_action_fingerprints": state.get("executed_action_fingerprints", []),
            "allowed_action_profiles": [
                {"action_kind": kind, "tool": tool, "profile": profile}
                for kind, tool, profile in sorted(ALLOWED_ACTION_PROFILES)
            ],
            "output_schema": {
                "test_plans": [
                    {
                        "target": "authorized target from context",
                        "strategy": "web_structure_validation",
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
        response = client.complete_json(
            prompt=json.dumps(prompt, ensure_ascii=False),
            system=STRATEGIST_SYSTEM_PROMPT,
            max_tokens=4096,
        )
        items = response.get("test_plans", [])
        if not isinstance(items, list):
            return []
        return self._coerce_llm_plans(items, state)

    def _coerce_llm_plans(self, items: list[dict[str, Any]], state: AutoFlowState) -> list[TestPlan]:
        allowed_targets = self._allowed_targets(state)
        plans: list[TestPlan] = []
        for item in items:
            target = canonical_target(str(item.get("target", "")))
            if not target or target not in allowed_targets:
                continue
            actions = self._coerce_llm_actions(item.get("actions", []), allowed_targets)
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
                    rationale=str(item.get("rationale") or "Generated by LLM discovery strategist."),
                    metadata={"source": "llm_strategist"},
                )
            )
        return plans

    def _coerce_llm_actions(self, items: Any, allowed_targets: set[str]) -> list[TestPlanAction]:
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
            if (action_kind, tool, profile) not in ALLOWED_ACTION_PROFILES:
                continue
            if not target or target not in allowed_targets:
                continue
            args = item.get("args") if isinstance(item.get("args"), dict) else {}
            if tool == "nikto":
                args = {"maxtime": str(args.get("maxtime", "60"))}
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
                    script_template=item.get("script_template") if tool == "script_runner" else None,
                    metadata={"source": "llm_strategist"},
                )
            )
        return actions

    def _allowed_targets(self, state: AutoFlowState) -> set[str]:
        targets: set[str] = set()
        for surface in state.get("attack_surfaces", []):
            if surface.get("target"):
                targets.add(canonical_target(str(surface["target"])))
            for entrypoint in surface.get("entrypoints", []):
                targets.add(canonical_target(str(entrypoint)))
        for item in state.get("web_recon", []):
            if item.get("target"):
                targets.add(canonical_target(str(item["target"])))
            for key in ("links", "interesting_paths"):
                for value in item.get(key, [])[:100]:
                    targets.add(canonical_target(str(value)))
            robots = item.get("robots") or {}
            for value in robots.get("interesting_paths", [])[:100]:
                targets.add(canonical_target(str(value)))
        for finding in state.get("findings", []):
            if finding.get("target"):
                targets.add(canonical_target(str(finding["target"])))
        return {target for target in targets if target}

    def _plans_from_attack_surfaces(self, surfaces: list[dict]) -> list[TestPlan]:
        plans: list[TestPlan] = []
        for surface in surfaces:
            if surface.get("surface_type") == "web_application":
                plans.append(self._web_fingerprint_plan(surface))
            elif surface.get("surface_type") == "network_service":
                plans.append(self._service_review_plan(surface))
        return plans

    def _plans_from_findings(self, findings: list[dict]) -> list[TestPlan]:
        # Candidate Finding 的漏洞验证策略由 ValidationAgent 负责。
        # Strategist 只继续做 discovery 推进，避免每个 finding 都重复触发 nikto/nuclei。
        return []

    def _plans_from_discovered_paths(self, web_recon: list[dict], findings: list[dict]) -> list[TestPlan]:
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
            self._web_recon_refresh_plan(canonical_target(target))
            for target in sorted({canonical_target(target) for target in targets})
            if target.startswith(("http://", "https://"))
        ]

    def _plans_from_web_recon(self, web_recon: list[dict]) -> list[TestPlan]:
        plans: list[TestPlan] = []
        for item in web_recon:
            target = item.get("target", "")
            if not target:
                continue
            has_structure = any(
                [
                    item.get("links"),
                    item.get("forms"),
                    item.get("scripts"),
                    item.get("interesting_paths"),
                    item.get("title"),
                ]
            )
            if has_structure:
                plans.append(self._web_structure_validation_plan(item))
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
            rationale="A web-like attack surface was found during research.",
            metadata={"attack_surface": surface},
        )

    def _service_review_plan(self, surface: dict) -> TestPlan:
        target = surface.get("target", "unknown")
        return TestPlan(
            target=target,
            strategy="service_exposure_review",
            angle="Exposed network service validation",
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            actions=[],
            rationale="The service is noted for review; no additional automatic tool is selected yet.",
            metadata={"attack_surface": surface},
        )

    def _web_validation_plan(self, finding: dict) -> TestPlan:
        target = finding.get("target", "unknown")
        return TestPlan(
            target=target,
            strategy="web_application_validation",
            angle="Web application configuration and safe template checks",
            risk_level=RiskLevel.MEDIUM,
            requires_approval=True,
            related_findings=[finding.get("id", "")],
            actions=[
                TestPlanAction(
                    name="Run basic web configuration checks",
                    action_kind="tool",
                    tool="nikto",
                    profile="basic_web_check",
                    target=target,
                    risk_level=RiskLevel.LOW,
                    requires_approval=False,
                    expected_impact="Read-only HTTP requests against the identified web service.",
                    rationale="Validate common web server misconfigurations and exposed files.",
                    args={"maxtime": "60"},
                ),
                self._nuclei_discovery_action(target),
                TestPlanAction(
                    name="Draft a safe custom HTTP header check",
                    action_kind="script",
                    tool="script_runner",
                    profile="security_headers_check",
                    target=target,
                    risk_level=RiskLevel.LOW,
                    requires_approval=False,
                    expected_impact="Read-only HTTP GET and header parsing.",
                    rationale="Script-based checks are useful when no packaged tool covers a narrow question.",
                    script_template="security_headers_check",
                ),
            ],
            rationale="A web application finding was observed, so continue with controlled validation angles.",
        )

    def _web_structure_validation_plan(self, web_recon: dict) -> TestPlan:
        target = web_recon.get("target", "unknown")
        forms = web_recon.get("forms", [])
        links = web_recon.get("links", [])
        scripts = web_recon.get("scripts", [])
        interesting_paths = web_recon.get("interesting_paths", [])
        nikto_maxtime = self._nikto_maxtime(web_recon)
        return TestPlan(
            target=target,
            strategy="web_structure_validation",
            angle="Validate discovered web structure, entrypoints, and common configuration weaknesses",
            risk_level=RiskLevel.MEDIUM,
            requires_approval=True,
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
                    rationale=(
                        "Web recon found a reachable application surface; validate common "
                        "misconfigurations and exposed files."
                    ),
                    args={"maxtime": str(nikto_maxtime)},
                    metadata={
                        "web_recon": web_recon,
                        "time_budget": {
                            "tool": "nikto",
                            "maxtime_seconds": nikto_maxtime,
                            "basis": self._web_recon_size(web_recon),
                        },
                    },
                ),
                self._nuclei_discovery_action(target, metadata={"web_recon": web_recon}),
                TestPlanAction(
                    name="Check security headers from discovered landing page",
                    action_kind="script",
                    tool="script_runner",
                    profile="security_headers_check",
                    target=target,
                    risk_level=RiskLevel.LOW,
                    requires_approval=False,
                    expected_impact="One read-only HTTP GET and header parsing.",
                    rationale="Security headers are relevant before deeper validation.",
                    script_template="security_headers_check",
                    metadata={"web_recon": web_recon},
                ),
            ],
            rationale=(
                f"Web recon observed title={web_recon.get('title')!r}, "
                f"{len(links)} links, {len(forms)} forms, {len(scripts)} scripts, "
                f"and {len(interesting_paths)} interesting paths."
            ),
            metadata={"web_recon": web_recon},
        )

    def _nuclei_discovery_action(self, target: str, metadata: dict | None = None) -> TestPlanAction:
        return TestPlanAction(
            name="Run discovery template checks",
            action_kind="tool",
            tool="nuclei",
            profile="discovery_all_severity",
            target=target,
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            expected_impact="Read-only nuclei discovery templates across all severities.",
            rationale=(
                "Collect candidate findings with safe read-only nuclei templates. "
                "Approval is deferred to later vulnerability validation or exploitation actions."
            ),
            metadata=metadata or {},
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
        """只保留 action 级审批项，避免 plan 级对象污染 Executor 队列。"""

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
        size = self._web_recon_size(web_recon)
        score = (
            size["links"]
            + size["interesting_paths"]
            + size["scripts"]
            + size["forms"] * 3
        )
        if score <= 10:
            return 60
        if score <= 50:
            return 90
        if score <= 150:
            return 120
        return 180

    def _web_recon_size(self, web_recon: dict) -> dict[str, int]:
        return {
            "links": len(web_recon.get("links", [])),
            "forms": len(web_recon.get("forms", [])),
            "scripts": len(web_recon.get("scripts", [])),
            "interesting_paths": len(web_recon.get("interesting_paths", [])),
        }
