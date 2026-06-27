from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autoflow.agents.tool_loop import AgentToolLoop
from autoflow.flows.models import FindingConfidence, ValidationResultStatus
from autoflow.graph.state import AutoFlowState
from autoflow.memory.agent_memory import AgentMemoryBuilder


VALIDATION_TOOL_PHASES = {"validation"}

VALIDATION_REACT_SYSTEM_PROMPT = """You are AutoFlow's ValidationReAct reasoner for an authorized lab security assessment.
Your job is to actively validate whether a candidate finding is confirmed, false positive, inconclusive, or needs more evidence.
Use function/tool calls to collect evidence when the current context is insufficient. Tool calls execute only inside the Docker tool container or AutoFlow built-in tools.
Stay strictly within authorized targets, paths, and findings already present in the provided state and memory.
Do not create destructive, persistence, evasion, lateral movement, or out-of-scope actions.
Prefer bounded read-only evidence collection unless the validation plan explicitly authorizes a stronger lab action.
Do not mark a finding confirmed unless you can cite concrete evidence from tool output, prior observations, or artifacts.
If more evidence is needed and a suitable tool is available, call the tool instead of ending early.
Return only JSON. Do not include markdown.
"""


@dataclass
class ValidationReasoningDecision:
    decision: ValidationResultStatus
    confidence: FindingConfidence = FindingConfidence.MEDIUM
    reasoning: str = ""
    impact: str = ""
    evidence: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    reproduction_steps: list[str] = field(default_factory=list)
    next_actions: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)


class ValidationReActReasoner:
    """LLM ReAct reasoner for vulnerability validation decisions."""

    def __init__(
        self,
        tool_loop: AgentToolLoop | None = None,
        memory_builder: AgentMemoryBuilder | None = None,
    ) -> None:
        self.tool_loop = tool_loop or AgentToolLoop(max_tool_rounds=5, max_tool_calls=8, max_tokens=1024)
        self.memory_builder = memory_builder or AgentMemoryBuilder()

    def reason(
        self,
        *,
        state: AutoFlowState,
        plan: dict[str, Any],
        action_results: list[dict[str, Any]],
    ) -> ValidationReasoningDecision:
        payload = self._payload(
            state=state,
            finding=self._plan_finding(plan),
            plan=plan,
            action_results=action_results,
            task="Decide whether the candidate finding is confirmed from current validation evidence.",
        )
        result = self._run(payload, state)
        return self._coerce_decision(result.final, result.tool_results, result.messages)

    def validate(
        self,
        *,
        state: AutoFlowState,
        finding: dict[str, Any],
        plan: dict[str, Any] | None = None,
        previous_results: list[dict[str, Any]] | None = None,
    ) -> ValidationReasoningDecision:
        payload = self._payload(
            state=state,
            finding=finding,
            plan=plan or {},
            action_results=previous_results or [],
            task=(
                "Actively validate this candidate finding. Call tools as needed, observe their results, "
                "and finish only when you can produce a grounded validation decision."
            ),
        )
        result = self._run(payload, state)
        return self._coerce_decision(result.final, result.tool_results, result.messages)

    def _run(self, payload: dict[str, Any], state: AutoFlowState):
        payload = self._with_validation_tool_manifest(payload)
        return self.tool_loop.run(
            system_prompt=VALIDATION_REACT_SYSTEM_PROMPT,
            user_payload=payload,
            state=state,
            final_repair_instruction=(
                "Return a JSON object with decision, confidence, reasoning, impact, evidence, "
                "missing_evidence, reproduction_steps, and next_actions."
            ),
            tools=self._openai_tools_for_payload(payload),
        )

    def _payload(
        self,
        *,
        state: AutoFlowState,
        finding: dict[str, Any],
        plan: dict[str, Any],
        action_results: list[dict[str, Any]],
        task: str,
    ) -> dict[str, Any]:
        payload = {
            "task": task,
            "mode": "react_validation",
            "memory_pack": self.memory_builder.build(state, persisted_memory=state.get("agent_memory")),
            "finding": finding,
            "validation_plan": plan,
            "action_results": self._compact_action_results(action_results),
            "recent_tool_observations": state.get("tool_observations", [])[-30:],
            "recent_executed_tasks": self._compact_executed_tasks(state.get("executed_tasks", [])[-30:]),
            "existing_validation_results": state.get("validation_results", [])[-30:],
            "known_targets": state.get("target_scope", []),
            "decision_rules": {
                "confirmed": "Evidence directly demonstrates the finding and impact with reproducible proof.",
                "false_positive": "Validation completed and evidence contradicts or does not reproduce the finding.",
                "inconclusive": "Evidence is incomplete, ambiguous, or execution failed without enough signal.",
                "need_more_evidence": "More bounded validation is required, but tool budget or scope prevents collecting it now.",
            },
            "validation_loop_guidance": [
                "Prefer curl, web_recon, script_runner, nuclei, nikto, and bounded bash inside the tool container.",
                "Use tools before finalizing if the finding has not already been reproduced by prior evidence.",
                "Tie every conclusion to specific response status, headers, body excerpts, tool signals, or artifacts.",
                "Return false_positive when the target is reachable but expected vulnerability indicators are absent.",
                "Return inconclusive when execution errors, redirects, auth gates, or ambiguous output prevent a conclusion.",
            ],
            "final_output_schema": {
                "decision": "confirmed|false_positive|inconclusive|need_more_evidence",
                "confidence": "low|medium|high",
                "reasoning": "short grounded explanation",
                "impact": "impact if confirmed, otherwise why impact is not established",
                "evidence": ["key evidence excerpts"],
                "missing_evidence": ["what is still missing"],
                "reproduction_steps": ["steps another tester can repeat"],
                "next_actions": [
                    {
                        "name": "optional follow-up action if decision is need_more_evidence",
                        "action_kind": "tool|script|shell|web_recon",
                        "tool": "curl|script_runner|bash_runner|web_recon|nuclei|nikto|whatweb",
                        "profile": "tool profile",
                        "target": "authorized target",
                        "risk_level": "low|medium|high|critical",
                        "requires_approval": False,
                        "args": {},
                        "rationale": "why this evidence is needed",
                    }
                ],
            },
        }
        return payload

    def _with_validation_tool_manifest(self, payload: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(payload)
        enriched["available_tool_manifest"] = self._prompt_manifest_for_payload(payload)
        enriched["tool_execution_boundary"] = {
            "containerized": True,
            "container_image": "autoflow-kali-tools",
            "host_shell_available_to_llm": False,
            "tools_filtered_for": "validation",
        }
        return enriched

    def _openai_tools_for_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        allowed = self._tool_names_for_payload(payload)
        return [
            function.openai_schema()
            for function in self.tool_loop.catalog.functions(VALIDATION_TOOL_PHASES)
            if function.name in allowed
        ]

    def _prompt_manifest_for_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        allowed = self._tool_names_for_payload(payload)
        manifest: list[dict[str, Any]] = []
        for function in self.tool_loop.catalog.functions(VALIDATION_TOOL_PHASES):
            if function.name not in allowed:
                continue
            metadata = function.metadata
            manifest.append(
                {
                    "function": function.name,
                    "tool": metadata.get("tool"),
                    "profile": metadata.get("profile"),
                    "kind": metadata.get("kind"),
                    "risk_level": metadata.get("risk_level"),
                    "purpose": function.description[:500],
                }
            )
        return manifest

    def _tool_names_for_payload(self, payload: dict[str, Any]) -> set[str]:
        finding = payload.get("finding") if isinstance(payload.get("finding"), dict) else {}
        metadata = finding.get("metadata") if isinstance(finding.get("metadata"), dict) else {}
        category = str(metadata.get("category") or "").lower()
        text = " ".join(
            str(value).lower()
            for value in [
                category,
                finding.get("title", ""),
                finding.get("description", ""),
                finding.get("target", ""),
            ]
        )
        base = {
            "read_agent_memory",
            "list_known_targets",
            "search_observations",
            "web_recon_fetch_page",
            "run_curl__get",
            "run_curl__get_with_headers",
        }
        if "cors" in text or "header" in text or "cache" in text:
            return {
                *base,
                "run_script__cors_probe",
                "run_script__security_headers_check",
            }
        if "api" in text or "/rest/" in text or "/api/" in text:
            return {
                *base,
                "run_script__api_endpoint_probe",
                "run_shell__bounded_bash",
            }
        if "debug" in text or "metrics" in text:
            return {
                *base,
                "run_script__debug_endpoint_probe",
                "run_shell__bounded_bash",
            }
        if "directory" in text or "listing" in text or "/ftp" in text:
            return {
                *base,
                "run_script__directory_listing_probe",
                "run_shell__bounded_bash",
            }
        if "config" in text or "package" in text or "secret" in text:
            return {
                *base,
                "run_script__public_config_probe",
                "run_shell__bounded_bash",
            }
        return {
            *base,
            "run_script__security_headers_check",
            "run_nuclei__discovery_all_severity",
        }

    def _coerce_decision(
        self,
        raw: dict[str, Any],
        tool_results: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> ValidationReasoningDecision:
        decision = self._decision(str(raw.get("decision", "inconclusive")))
        confidence = self._confidence(str(raw.get("confidence", "medium")))
        return ValidationReasoningDecision(
            decision=decision,
            confidence=confidence,
            reasoning=str(raw.get("reasoning") or ""),
            impact=str(raw.get("impact") or ""),
            evidence=self._string_list(raw.get("evidence")),
            missing_evidence=self._string_list(raw.get("missing_evidence")),
            reproduction_steps=self._string_list(raw.get("reproduction_steps")),
            next_actions=[item for item in raw.get("next_actions", []) if isinstance(item, dict)]
            if isinstance(raw.get("next_actions"), list)
            else [],
            raw=raw,
            tool_results=tool_results,
            messages=messages,
        )

    def _decision(self, value: str) -> ValidationResultStatus:
        normalized = value.lower().strip()
        mapping = {
            "confirmed": ValidationResultStatus.VALIDATED,
            "validated": ValidationResultStatus.VALIDATED,
            "false_positive": ValidationResultStatus.FALSE_POSITIVE,
            "false positive": ValidationResultStatus.FALSE_POSITIVE,
            "inconclusive": ValidationResultStatus.INCONCLUSIVE,
            "need_more_evidence": ValidationResultStatus.INCONCLUSIVE,
            "needs_more_evidence": ValidationResultStatus.INCONCLUSIVE,
        }
        return mapping.get(normalized, ValidationResultStatus.INCONCLUSIVE)

    def _confidence(self, value: str) -> FindingConfidence:
        try:
            return FindingConfidence(value.lower().strip())
        except ValueError:
            return FindingConfidence.MEDIUM

    def _compact_action_results(self, action_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact = []
        for item in action_results[-20:]:
            compact.append(
                {
                    "action_id": item.get("action_id"),
                    "status": item.get("status"),
                    "summary": self._trim(item.get("summary", ""), 1000),
                    "error": self._trim(item.get("error", ""), 1000),
                    "stdout": self._trim(item.get("stdout", ""), 2500),
                    "stderr": self._trim(item.get("stderr", ""), 1200),
                    "artifact_id": item.get("artifact_id"),
                }
            )
        return compact

    def _compact_executed_tasks(self, executed_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact = []
        for item in executed_tasks:
            task = item.get("task") if isinstance(item.get("task"), dict) else {}
            compact.append(
                {
                    "action_id": item.get("action_id"),
                    "status": item.get("status"),
                    "tool": task.get("tool"),
                    "profile": task.get("profile"),
                    "target": task.get("target"),
                    "summary": self._trim(item.get("summary", ""), 1000),
                    "error": self._trim(item.get("error", ""), 1000),
                }
            )
        return compact

    def _plan_finding(self, plan: dict[str, Any]) -> dict[str, Any]:
        metadata = plan.get("metadata") if isinstance(plan.get("metadata"), dict) else {}
        finding = metadata.get("finding")
        return finding if isinstance(finding, dict) else {}

    def _string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item).strip()][:20]

    def _trim(self, value: object, limit: int) -> str:
        text = str(value or "")
        return text if len(text) <= limit else text[: limit - 3] + "..."
