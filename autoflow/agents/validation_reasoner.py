from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from autoflow.agents.tool_loop import AgentToolLoop
from autoflow.flows.models import FindingConfidence, ValidationResultStatus
from autoflow.graph.state import AutoFlowState
from autoflow.memory.agent_memory import AgentMemoryBuilder


VALIDATION_TOOL_PHASES = {"validation"}

VALIDATION_REACT_SYSTEM_PROMPT = """You are AutoFlow's ValidationReAct reasoner for an authorized lab security assessment.
Your job is to actively validate whether a candidate finding is confirmed, false positive, inconclusive, or needs more evidence.
Use function/tool calls to collect evidence when the current context is insufficient. Tool calls execute only inside the Docker tool container or AutoFlow built-in tools.
You may choose any provided validation tool yourself based on the finding, memory, and previous tool results.
When existing tools do not answer the exact validation question, write and run a custom validation Python script through run_script__custom_validation.
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
        memory_pack = self.memory_builder.build(state, persisted_memory=state.get("agent_memory"))
        payload = {
            "task": task,
            "mode": "react_validation",
            "memory_pack": self._compact_memory_pack(memory_pack),
            "finding": finding,
            "validation_plan": plan,
            "validation_context": self._validation_context(state, finding),
            "action_results": self._compact_action_results(action_results),
            "recent_tool_observations": self._compact_observations(state.get("tool_observations", [])[-30:]),
            "related_tool_observations": self._related_observations(state, finding),
            "recent_executed_tasks": self._compact_executed_tasks(state.get("executed_tasks", [])[-30:]),
            "existing_validation_results": state.get("validation_results", [])[-30:],
            "known_targets": sorted(self._known_targets(state)),
            "recommended_tools": sorted(self._recommended_tool_names_for_payload({"finding": finding})),
            "decision_rules": {
                "confirmed": "Evidence directly demonstrates the finding and impact with reproducible proof.",
                "false_positive": "Validation completed and evidence contradicts or does not reproduce the finding.",
                "inconclusive": "Evidence is incomplete, ambiguous, or execution failed without enough signal.",
                "need_more_evidence": "More bounded validation is required, but tool budget or scope prevents collecting it now.",
            },
            "validation_loop_guidance": [
                "Select tools yourself from available_tool_manifest; recommended_tools are hints, not a hard whitelist.",
                "Use curl or web_recon to confirm reachability, headers, redirects, body excerpts, forms, and endpoint behavior.",
                "Use script_runner templates for common validation questions such as CORS, API exposure, directory listing, public config, debug endpoint, and security headers.",
                "Use run_script__custom_validation when the validation requires custom logic over multiple requests or response comparisons.",
                "Use run_shell__bounded_bash for compact container-only curl/grep/jq/python pipelines when a shell is more direct.",
                "Use sqlmap only for authorized URLs with query parameters and prior injection evidence.",
                "Use hydra/medusa only when a specific known credential pair is present in memory or evidence; never invent wordlists.",
                "Use tools before finalizing if the finding has not already been reproduced by prior evidence.",
                "Tie every conclusion to specific response status, headers, body excerpts, tool signals, or artifacts.",
                "Return false_positive when the target is reachable but expected vulnerability indicators are absent.",
                "Return inconclusive when execution errors, redirects, auth gates, or ambiguous output prevent a conclusion.",
            ],
            "custom_script_contract": {
                "tool": "run_script__custom_validation",
                "execution_boundary": "Docker container only, never host shell",
                "provided_variables": ["TARGET", "TARGET_SCOPE", "ARTIFACT_DIR"],
                "preferred_imports": ["json", "re", "ssl", "socket", "sys", "time", "urllib"],
                "output": "print one concise JSON object with target, probe, status, evidence, and conclusion fields",
                "timeouts": "use short network timeouts, normally <= 10 seconds per request",
                "scope": "use TARGET or discovered authorized paths only",
            },
            "required_final_fields": [
                "decision",
                "confidence",
                "reasoning",
                "impact",
                "evidence",
                "missing_evidence",
                "reproduction_steps",
                "next_actions",
            ],
            "json_contract": {
                "required_top_level_fields": [
                    "decision",
                    "confidence",
                    "reasoning",
                    "impact",
                    "evidence",
                    "missing_evidence",
                    "reproduction_steps",
                    "next_actions",
                ],
                "return_only_json_object": True,
                "no_markdown": True,
                "arrays_may_be_empty": True,
                "decision_enum": ["confirmed", "false_positive", "inconclusive", "need_more_evidence"],
                "confidence_enum": ["low", "medium", "high"],
            },
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
                        "tool": "curl|script_runner|bash_runner|web_recon|nuclei|nikto|whatweb|sqlmap|hydra|medusa",
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

    def _compact_memory_pack(self, memory: dict[str, Any]) -> dict[str, Any]:
        return {
            "project_id": memory.get("project_id"),
            "flow_id": memory.get("flow_id"),
            "target_scope": memory.get("target_scope", [])[:20],
            "user_prompt": self._trim(memory.get("user_prompt", ""), 800),
            "rules_of_engagement": memory.get("rules_of_engagement", {}),
            "memory_counts": memory.get("memory_counts", {}),
            "known_assets": [
                {
                    "ip": item.get("ip"),
                    "hostnames": item.get("hostnames", [])[:8],
                    "ports": item.get("ports", [])[:20],
                }
                for item in memory.get("known_assets", [])[-20:]
                if isinstance(item, dict)
            ],
            "web_context": self._compact_web_context(memory.get("web_context", [])[-20:]),
            "candidate_findings": [
                self._compact_finding(item)
                for item in memory.get("candidate_findings", [])[-30:]
                if isinstance(item, dict)
            ],
            "validation_plans": [
                {
                    "id": item.get("id"),
                    "finding_id": item.get("finding_id"),
                    "target": item.get("target"),
                    "objective": self._trim(item.get("objective", ""), 500),
                    "risk_level": item.get("risk_level"),
                    "status": item.get("status"),
                }
                for item in memory.get("validation_plans", [])[-30:]
                if isinstance(item, dict)
            ],
            "tool_observations": self._compact_observations(memory.get("tool_observations", [])[-30:]),
            "executed_actions": memory.get("executed_actions", [])[-20:],
            "failed_actions": memory.get("failed_actions", [])[-20:],
            "flow_memories": [
                {
                    "kind": item.get("kind"),
                    "source": item.get("source"),
                    "content": self._trim(item.get("content", ""), 700),
                }
                for item in memory.get("flow_memories", [])[-20:]
                if isinstance(item, dict)
            ],
        }

    def _validation_context(self, state: AutoFlowState, finding: dict[str, Any]) -> dict[str, Any]:
        target = str(finding.get("target") or "")
        return {
            "finding_summary": self._compact_finding(finding),
            "target": target,
            "target_host": self._host(target),
            "related_web_recon": self._related_web_context(state, finding),
            "related_observations": self._related_observations(state, finding),
            "related_validation_results": [
                item
                for item in state.get("validation_results", [])[-20:]
                if item.get("finding_id") == finding.get("id")
            ],
            "candidate_paths": self._candidate_paths(state, finding),
        }

    def _compact_finding(self, finding: dict[str, Any]) -> dict[str, Any]:
        metadata = finding.get("metadata") if isinstance(finding.get("metadata"), dict) else {}
        return {
            "id": finding.get("id"),
            "title": finding.get("title"),
            "target": finding.get("target"),
            "severity": finding.get("severity"),
            "confidence": finding.get("confidence"),
            "status": finding.get("status"),
            "category": metadata.get("category"),
            "description": self._trim(finding.get("description", ""), 900),
            "evidence": [self._trim(item, 700) for item in finding.get("evidence", [])[:8]],
            "recommendation": self._trim(finding.get("recommendation", ""), 500),
        }

    def _compact_web_context(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "target": item.get("target"),
                "status_code": item.get("status_code"),
                "title": item.get("title"),
                "links": item.get("links", [])[:30],
                "forms": [
                    {
                        "action": form.get("action"),
                        "method": form.get("method"),
                        "inputs": len(form.get("inputs", [])),
                    }
                    for form in item.get("forms", [])[:8]
                    if isinstance(form, dict)
                ],
                "scripts": item.get("scripts", [])[:20],
                "interesting_paths": item.get("interesting_paths", [])[:30],
                "robots": item.get("robots", {}),
                "error": item.get("error", ""),
            }
            for item in items
            if isinstance(item, dict)
        ]

    def _compact_observations(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact = []
        for item in observations:
            if not isinstance(item, dict):
                continue
            compact.append(
                {
                    "tool": item.get("tool"),
                    "profile": item.get("profile"),
                    "target": item.get("target"),
                    "status": item.get("status"),
                    "summary": self._trim(item.get("summary", ""), 700),
                    "signals": [
                        {
                            "kind": signal.get("kind"),
                            "name": signal.get("name"),
                            "severity": signal.get("severity"),
                            "target": signal.get("target"),
                            "evidence": self._trim(signal.get("evidence", ""), 500),
                        }
                        for signal in item.get("signals", [])[:12]
                        if isinstance(signal, dict)
                    ],
                }
            )
        return compact

    def _related_web_context(self, state: AutoFlowState, finding: dict[str, Any]) -> list[dict[str, Any]]:
        target = str(finding.get("target") or "")
        target_host = self._host(target)
        related = []
        for item in state.get("web_recon", []):
            item_target = str(item.get("target") or "")
            if item_target == target or (target_host and self._host(item_target) == target_host):
                related.append(item)
        return self._compact_web_context(related[-12:])

    def _related_observations(self, state: AutoFlowState, finding: dict[str, Any]) -> list[dict[str, Any]]:
        target = str(finding.get("target") or "")
        target_host = self._host(target)
        keywords = self._finding_keywords(finding)
        related = []
        for item in state.get("tool_observations", []):
            text = " ".join(
                [
                    str(item.get("target", "")),
                    str(item.get("summary", "")),
                    " ".join(str(signal.get("name", "")) for signal in item.get("signals", []) if isinstance(signal, dict)),
                    " ".join(str(signal.get("kind", "")) for signal in item.get("signals", []) if isinstance(signal, dict)),
                ]
            ).lower()
            item_host = self._host(str(item.get("target") or ""))
            if (target_host and item_host == target_host) or any(keyword in text for keyword in keywords):
                related.append(item)
        return self._compact_observations(related[-20:])

    def _candidate_paths(self, state: AutoFlowState, finding: dict[str, Any]) -> list[str]:
        paths = []
        for item in self._related_web_context(state, finding):
            paths.extend(str(value) for value in item.get("links", [])[:20])
            paths.extend(str(value) for value in item.get("interesting_paths", [])[:20])
            robots = item.get("robots") if isinstance(item.get("robots"), dict) else {}
            paths.extend(str(value) for value in robots.get("interesting_paths", [])[:20])
        for evidence in finding.get("evidence", [])[:8]:
            for token in str(evidence).split():
                if token.startswith(("http://", "https://", "/")):
                    paths.append(token.strip(".,;)'\""))
        return list(dict.fromkeys(path for path in paths if path))[:80]

    def _known_targets(self, state: AutoFlowState) -> set[str]:
        targets = {str(value) for value in state.get("target_scope", []) if value}
        for item in state.get("assets", []):
            host = str(item.get("ip", ""))
            if host:
                targets.add(host)
            for port in item.get("ports", []):
                port_number = port.get("port")
                if port_number:
                    targets.add(f"{host}:{port_number}")
                    scheme = "https" if int(port_number) in {443, 8443} else "http"
                    targets.add(f"{scheme}://{host}:{port_number}")
        for collection_name in ("web_recon", "attack_surfaces", "findings"):
            for item in state.get(collection_name, []):
                if item.get("target"):
                    targets.add(str(item["target"]))
                for value in item.get("entrypoints", []):
                    targets.add(str(value))
        return {target for target in targets if target}

    def _finding_keywords(self, finding: dict[str, Any]) -> set[str]:
        metadata = finding.get("metadata") if isinstance(finding.get("metadata"), dict) else {}
        values = [
            finding.get("title", ""),
            finding.get("description", ""),
            metadata.get("category", ""),
            *finding.get("evidence", [])[:5],
        ]
        keywords = set()
        for value in values:
            for token in str(value).lower().replace("/", " ").replace("_", " ").replace("-", " ").split():
                if len(token) >= 4:
                    keywords.add(token[:40])
        return keywords

    def _host(self, value: str) -> str:
        parsed = urlparse(value if "://" in value else f"//{value}")
        return parsed.hostname or ""

    def _tool_names_for_payload(self, payload: dict[str, Any]) -> set[str]:
        names = {function.name for function in self.tool_loop.catalog.functions(VALIDATION_TOOL_PHASES)}
        if names:
            return names
        return self._recommended_tool_names_for_payload(payload)

    def _recommended_tool_names_for_payload(self, payload: dict[str, Any]) -> set[str]:
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
            "run_curl__head",
            "run_script__custom_validation",
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
                "run_script__custom_validation",
            }
        if "sql" in text or "sqli" in text or "injection" in text:
            return {
                *base,
                "run_sqlmap__basic_get_param_check",
                "run_shell__bounded_bash",
                "run_script__custom_validation",
            }
        if "credential" in text or "login" in text or "default password" in text or "password" in text:
            return {
                *base,
                "run_hydra__single_credential_check",
                "run_medusa__single_credential_check",
                "run_shell__bounded_bash",
                "run_script__custom_validation",
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
            "run_shell__bounded_bash",
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
