from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from autoflow.agents.tool_loop import AgentToolLoop
from autoflow.flows.models import FindingConfidence, ValidationResultStatus
from autoflow.graph.state import AutoFlowState
from autoflow.memory.agent_memory import AgentMemoryBuilder
from autoflow.memory.compressor import MemoryCompressor


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
Do not confirm suspected exploitability from weak signals. Category-specific proof is required before a finding is validated.
If more evidence is needed and a suitable tool is available, call the tool instead of ending early.
Return only JSON. Do not include markdown.
"""


VALIDATION_EVIDENCE_REQUIREMENTS: dict[str, list[str]] = {
    "generic": [
        "At least one concrete tool observation, artifact, or response excerpt must directly support the claim.",
        "A confirmed result must include reproducible steps and specific status/header/body/tool output evidence.",
        "Scanner labels alone are not enough when the output does not show why the issue is exploitable or exposed.",
    ],
    "api_exposure": [
        "Show the exact endpoint returns 2xx without authentication or with the tested anonymous context.",
        "Quote response content type, representative JSON keys/body sample, or sensitive/business data indicators.",
        "Do not claim authorization bypass unless authenticated or cross-user comparisons prove it.",
    ],
    "cors": [
        "Quote the exact Access-Control-Allow-Origin and Access-Control-Allow-Credentials behavior from a live response.",
        "If claiming exploitable sensitive-data CORS, also show a sensitive endpoint body is readable under that CORS policy.",
    ],
    "security_header": [
        "Quote live response headers or header-check output showing the exact missing or weak header.",
        "Describe the affected page/endpoint; do not claim direct exploitation from a missing hardening header alone.",
    ],
    "directory_listing": [
        "Show an Index of/directory listing response or parsed href entries from the target path.",
        "Impact requires listed sensitive files such as backups, config, keys, logs, databases, archives, or source files.",
    ],
    "public_config": [
        "Show the public file is reachable with 2xx and quote config/dependency/endpoint/secret-like keys or body excerpts.",
        "Do not claim secret disclosure unless an actual secret-like value or sensitive internal metadata is present.",
    ],
    "debug_endpoint": [
        "Show the debug/metrics endpoint returns 2xx and quote runtime, stack, env, trace, metric, or secret-like content.",
        "Health checks or generic status pages are not enough for a confirmed debug information exposure.",
    ],
    "sql_injection": [
        "Show sqlmap or equivalent tool output identifying an injectable parameter, or quote reproducible payload behavior.",
        "Acceptable proof includes DB error disclosure, boolean/content differential, time delay differential, or extracted DB metadata.",
        "A quote character returning fewer rows, a 200 response, or a scanner suspicion alone is not enough.",
    ],
    "xss": [
        "Show the payload reflected/stored in a response or DOM sink with executable context evidence.",
        "A parameter that accepts input, or HTML/JS technology fingerprinting, is not enough.",
    ],
    "idor_authz": [
        "Show unauthorized access to another object or role by comparing at least two identities or auth/no-auth contexts.",
        "A 2xx response from a public endpoint alone is not enough.",
    ],
    "ssrf": [
        "Show a controlled callback, metadata endpoint access, internal-service response, or equivalent server-side fetch evidence.",
        "A URL parameter or fetch-like feature alone is not enough.",
    ],
    "file_upload": [
        "Show upload success and server-side retrieval, execution, unsafe storage, or content-type bypass evidence.",
        "A visible upload form alone is not enough.",
    ],
    "credential_access": [
        "Show a successful authentication response, session cookie, token, account context, or equivalent proof for the tested credential.",
        "The existence of a login form or a known default credential list is not enough.",
    ],
    "network_share": [
        "Show anonymous or tested-credential share enumeration, readable file listing, or file retrieval evidence.",
        "Open SMB/RPC ports alone are not enough.",
    ],
    "source_artifact": [
        "Show static analysis output with file path, rule id/name, severity, and code or secret evidence.",
        "Repository or artifact availability alone is not enough.",
    ],
}


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
        memory_compressor: MemoryCompressor | None = None,
    ) -> None:
        self.tool_loop = tool_loop or AgentToolLoop(max_tool_rounds=5, max_tool_calls=8, max_tokens=1024)
        self.memory_builder = memory_builder or AgentMemoryBuilder()
        self.memory_compressor = memory_compressor or MemoryCompressor()

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
        return self._coerce_decision(
            result.final,
            result.tool_results,
            result.messages,
            finding=self._plan_finding(plan),
        )

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
        return self._coerce_decision(result.final, result.tool_results, result.messages, finding=finding)

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
        validation_memory_view = self.memory_compressor.build_for_agent(
            state,
            agent_name="validation_react",
            focus={
                "finding": finding,
                "finding_id": finding.get("id"),
                "target": finding.get("target"),
                "category": (finding.get("metadata") or {}).get("category")
                if isinstance(finding.get("metadata"), dict)
                else "",
            },
            base_memory=memory_pack,
            validation_plan=plan,
            recent_tool_results=action_results,
        )
        self._record_memory_budget(state, validation_memory_view)
        payload = {
            "task": task,
            "mode": "react_validation",
            "memory_pack": validation_memory_view.get("base_memory_summary", self._compact_memory_pack(memory_pack)),
            "validation_memory_view": validation_memory_view,
            "finding": finding,
            "validation_plan": plan,
            "validation_context": self._validation_context(state, finding),
            "action_results": self._compact_action_results(action_results),
            "recent_tool_observations": validation_memory_view.get("related_observations", [])[:12],
            "related_tool_observations": validation_memory_view.get("related_observations", []),
            "recent_executed_tasks": self._compact_executed_tasks(state.get("executed_tasks", [])[-12:]),
            "existing_validation_results": validation_memory_view.get("related_validation_history", []),
            "known_targets": sorted(self._known_targets(state)),
            "recommended_tools": sorted(self._recommended_tool_names_for_payload({"finding": finding})),
            "decision_rules": {
                "confirmed": "Evidence directly demonstrates the finding and impact with reproducible proof.",
                "false_positive": "Validation completed and evidence contradicts or does not reproduce the finding.",
                "inconclusive": "Evidence is incomplete, ambiguous, or execution failed without enough signal.",
                "need_more_evidence": "More bounded validation is required, but tool budget or scope prevents collecting it now.",
            },
            "evidence_quality_gate": {
                "purpose": "A confirmed validation must meet the category-specific evidence requirements below.",
                "downgrade_rule": (
                    "If evidence does not satisfy the relevant category, return inconclusive and list the missing proof. "
                    "Separate observed exposure from suspected exploitability."
                ),
                "category_requirements": VALIDATION_EVIDENCE_REQUIREMENTS,
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
                "Do not validate a vulnerability class just because a scanner or heuristic suggested it; reproduce the class-specific behavior.",
                "When a response proves only exposure, validate only that exposure and keep stronger exploit claims inconclusive.",
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

    def _record_memory_budget(self, state: AutoFlowState, memory_view: dict[str, Any]) -> None:
        report = memory_view.get("budget_report")
        if not isinstance(report, dict):
            return
        reports = state.get("memory_budget_reports", [])
        if not isinstance(reports, list):
            reports = []
        reports.append(report)
        state["memory_budget_reports"] = reports[-50:]
        state["last_memory_budget"] = report

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
        finding: dict[str, Any] | None = None,
    ) -> ValidationReasoningDecision:
        raw = dict(raw) if isinstance(raw, dict) else {}
        decision = self._decision(str(raw.get("decision", "inconclusive")))
        confidence = self._confidence(str(raw.get("confidence", "medium")))
        reasoning = str(raw.get("reasoning") or "")
        evidence = self._string_list(raw.get("evidence"))
        missing_evidence = self._string_list(raw.get("missing_evidence"))
        reproduction_steps = self._string_list(raw.get("reproduction_steps"))
        next_actions = [item for item in raw.get("next_actions", []) if isinstance(item, dict)] if isinstance(raw.get("next_actions"), list) else []

        if decision == ValidationResultStatus.VALIDATED:
            passed, gate_category, gate_missing = self._validated_evidence_gate(
                finding=finding or {},
                raw=raw,
                tool_results=tool_results,
                evidence=evidence,
                reproduction_steps=reproduction_steps,
            )
            raw["evidence_gate"] = {
                "passed": passed,
                "category": gate_category,
                "missing": gate_missing,
            }
            if not passed:
                decision = ValidationResultStatus.INCONCLUSIVE
                if confidence == FindingConfidence.HIGH:
                    confidence = FindingConfidence.MEDIUM
                missing_evidence = self._dedupe_text([*missing_evidence, *gate_missing])
                gate_reason = (
                    f"AutoFlow evidence gate downgraded this validation to inconclusive because "
                    f"{gate_category} proof is incomplete: {'; '.join(gate_missing)}."
                )
                reasoning = f"{gate_reason} Original reasoning: {reasoning}" if reasoning else gate_reason

        return ValidationReasoningDecision(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            impact=str(raw.get("impact") or ""),
            evidence=evidence,
            missing_evidence=missing_evidence,
            reproduction_steps=reproduction_steps,
            next_actions=next_actions,
            raw=raw,
            tool_results=tool_results,
            messages=messages,
        )

    def _validated_evidence_gate(
        self,
        *,
        finding: dict[str, Any],
        raw: dict[str, Any],
        tool_results: list[dict[str, Any]],
        evidence: list[str],
        reproduction_steps: list[str],
    ) -> tuple[bool, str, list[str]]:
        category = self._validation_category(finding, raw)
        text = self._evidence_blob(finding=finding, raw=raw, tool_results=tool_results)
        missing: list[str] = []

        if not self._has_concrete_tool_evidence(text):
            missing.append("缺少具体工具输出、HTTP 响应片段、artifact 或可复现实验结果")
        if not evidence:
            missing.append("缺少可引用的 evidence 条目")
        if not reproduction_steps:
            missing.append("缺少可复现步骤")

        if category == "api_exposure":
            if not self._has_success_status(text):
                missing.append("API 暴露需要证明目标端点返回 2xx/成功状态")
            if not self._has_auth_context(text):
                missing.append("API 暴露需要证明是在匿名、无凭据或明确测试身份下访问")
            if not self._has_body_data(text):
                missing.append("API 暴露需要响应体样本、JSON key、业务数据或敏感字段证据")
        elif category == "cors":
            if not self._has_any(text, ["access-control-allow-origin", "acao"]):
                missing.append("CORS 需要原始 Access-Control-Allow-Origin 响应头证据")
            if not self._has_any(text, ["*", "wildcard", "reflect", "reflected", "null"]):
                missing.append("CORS 需要证明通配、反射或异常 Origin 行为")
            if self._has_positive_claim(text, ["sensitive data", "credentialed cors", "credentials readable"]):
                if not self._has_body_data(text):
                    missing.append("CORS 敏感数据影响需要可读取响应体或敏感字段证据")
        elif category == "security_header":
            if not self._has_any(
                text,
                [
                    "content-security-policy",
                    "strict-transport-security",
                    "x-frame-options",
                    "x-content-type-options",
                    "referrer-policy",
                    "permissions-policy",
                    "feature-policy",
                    "cache-control",
                ],
            ):
                missing.append("安全头问题需要明确的响应头名称")
            if not self._has_any(text, ["missing", "absent", "not present", "not set", "weak", "unsafe", "no-store"]):
                missing.append("安全头问题需要证明该响应头缺失或配置薄弱")
        elif category == "directory_listing":
            if not self._has_any(text, ["index of", "directory listing", "href=", "[entries]", "listed files"]):
                missing.append("目录 listing 需要 Index of、目录列表页面或解析出的 href/文件条目")
            if self._has_positive_claim(text, ["sensitive file", "secret", "credential", "database", "backup"]):
                if not self._has_any(text, [".bak", ".backup", ".old", ".zip", ".tar", ".gz", ".db", ".sqlite", ".key", ".pem", ".log", ".env", "config"]):
                    missing.append("目录 listing 的敏感影响需要具体敏感文件名或文件类型证据")
        elif category == "public_config":
            if not self._has_success_status(text):
                missing.append("公开配置文件需要证明文件可 2xx 访问")
            if not self._has_any(text, ["package.json", ".env", "config", "manifest", "swagger", "openapi", "robots.txt", "sitemap.xml"]):
                missing.append("公开配置文件需要具体文件名或配置类型")
            if not self._has_any(text, ["dependencies", "version", "endpoint", "secret", "token", "api_key", "apikey", "password", "redis", "mongodb", "jwt"]):
                missing.append("公开配置文件需要配置、依赖、端点或 secret-like 字段证据")
        elif category == "debug_endpoint":
            if not self._has_success_status(text):
                missing.append("debug endpoint 需要证明端点返回 2xx/成功状态")
            if not self._has_any(text, ["stack", "trace", "exception", "env", "process", "heap", "memory", "uptime", "metrics", "prometheus", "secret", "token"]):
                missing.append("debug endpoint 需要运行时、堆栈、指标、环境或敏感字段证据")
        elif category == "sql_injection":
            if not self._has_sql_injection_proof(text):
                missing.append("SQL 注入需要 sqlmap/等价工具确认可注入参数，或 DB 错误、布尔差异、时间差异、UNION/元数据提取证据")
        elif category == "xss":
            if not self._has_any(text, ["<script", "onerror=", "onload=", "alert(", "payload"]):
                missing.append("XSS 需要具体 payload")
            if not self._has_any(text, ["reflected", "stored", "executed", "dom sink", "html context", "script context"]):
                missing.append("XSS 需要反射、存储或 DOM sink 的执行上下文证据")
        elif category == "idor_authz":
            if not self._has_any(text, ["user a", "user b", "two users", "another user", "cross-user", "unauthorized", "forbidden bypass", "object id", "owner"]):
                missing.append("IDOR/鉴权绕过需要跨用户、跨对象或有无认证对比证据")
        elif category == "ssrf":
            if not self._has_any(text, ["interactsh", "callback", "dns hit", "oob", "169.254.169.254", "metadata", "internal service", "server-side request"]):
                missing.append("SSRF 需要回连、metadata、内网服务响应或等价服务端请求证据")
        elif category == "file_upload":
            if not self._has_any(text, ["upload success", "uploaded", "201", "stored", "retrieved", "downloaded", "executed", "content-type bypass"]):
                missing.append("文件上传漏洞需要上传成功和服务端存储、访问、执行或绕过证据")
        elif category == "credential_access":
            if not self._has_any(text, ["login successful", "authenticated", "set-cookie", "session", "jwt", "token", "dashboard", "whoami"]):
                missing.append("弱口令/默认口令需要成功认证响应、session、token 或账号上下文证据")
        elif category == "network_share":
            if not self._has_any(text, ["anonymous", "share", "smb", "readable", "smbmap", "smbclient", "enum4linux", "file listing"]):
                missing.append("网络共享问题需要匿名/测试凭据枚举、可读共享或文件列表证据")
        elif category == "source_artifact":
            if not self._has_any(text, ["semgrep", "bandit", "trivy", "gitleaks", "rule", "severity", "file", "line"]):
                missing.append("源码或制品审计需要工具规则、严重级别、文件路径和证据片段")
        else:
            if not (self._has_success_status(text) or self._has_any(text, ["confirmed by", "vulnerable", "artifact", "tool output", "response body"])):
                missing.append("通用漏洞确认需要成功响应、工具确认、artifact 或响应体证据")

        missing.extend(self._unsupported_claim_requirements(category, text))
        missing = self._dedupe_text(missing)
        return not missing, category, missing

    def _validation_category(self, finding: dict[str, Any], raw: dict[str, Any]) -> str:
        metadata = finding.get("metadata") if isinstance(finding.get("metadata"), dict) else {}
        explicit = str(metadata.get("category") or "").lower()
        explicit_map = [
            ("api_exposure", ["api_exposure", "api exposure"]),
            ("cors", ["cors", "cors_wildcard"]),
            ("security_header", ["missing_security_header", "security_header", "weak_cache_control", "informational_header"]),
            ("directory_listing", ["directory_listing", "directory listing"]),
            ("public_config", ["public_config", "public config", "package.json", "robots_txt", "robots.txt"]),
            ("debug_endpoint", ["debug_endpoint", "debug endpoint", "metrics"]),
            ("sql_injection", ["sql_injection", "sqli", "sql injection"]),
            ("xss", ["xss", "cross_site_scripting", "cross-site scripting"]),
            ("idor_authz", ["idor", "auth_bypass", "authorization", "access_control"]),
            ("ssrf", ["ssrf"]),
            ("file_upload", ["file_upload", "file upload"]),
            ("credential_access", ["credential", "default_password", "weak_password", "bruteforce"]),
            ("network_share", ["smb", "network_share", "share"]),
            ("source_artifact", ["semgrep", "bandit", "trivy", "gitleaks", "source_artifact"]),
        ]
        for category, aliases in explicit_map:
            if any(alias in explicit for alias in aliases):
                return category

        text = self._evidence_blob(finding=finding, raw=raw, tool_results=[])
        inferred_map = [
            ("sql_injection", ["sql injection", "sqli", "sqlmap", "injectable parameter"]),
            ("xss", ["xss", "cross-site scripting", "reflected payload", "<script"]),
            ("idor_authz", ["idor", "authorization bypass", "access control", "another user"]),
            ("ssrf", ["ssrf", "server-side request", "metadata endpoint"]),
            ("file_upload", ["file upload", "upload bypass"]),
            ("credential_access", ["default credential", "weak password", "login successful"]),
            ("network_share", ["smb", "smbclient", "smbmap", "enum4linux"]),
            ("debug_endpoint", ["debug endpoint", "metrics endpoint", "stack trace"]),
            ("directory_listing", ["directory listing", "index of"]),
            ("public_config", ["package.json", ".env", "public config", "swagger", "openapi"]),
            ("cors", ["cors", "access-control-allow-origin"]),
            ("security_header", ["security header", "content-security-policy", "x-frame-options", "strict-transport-security"]),
            ("api_exposure", ["api", "/api/", "/rest/", "json endpoint"]),
            ("source_artifact", ["semgrep", "bandit", "trivy", "gitleaks"]),
        ]
        for category, aliases in inferred_map:
            if any(alias in text for alias in aliases):
                return category
        return "generic"

    def _unsupported_claim_requirements(self, category: str, text: str) -> list[str]:
        checks = [
            (
                "sql_injection",
                ["sql injection", "sqli", "injectable"],
                self._has_sql_injection_proof,
                "最终结论提到了 SQL 注入，但缺少可注入参数、DB 错误、布尔/时间差异或工具确认",
            ),
            (
                "xss",
                ["xss", "cross-site scripting"],
                lambda value: self._has_any(value, ["<script", "payload", "reflected", "stored", "dom sink"]),
                "最终结论提到了 XSS，但缺少 payload 与反射/存储/DOM sink 证据",
            ),
            (
                "idor_authz",
                ["idor", "authorization bypass", "auth bypass"],
                lambda value: self._has_any(value, ["user a", "user b", "another user", "cross-user", "unauthorized"]),
                "最终结论提到了 IDOR/鉴权绕过，但缺少跨用户或未授权访问对比证据",
            ),
            (
                "ssrf",
                ["ssrf", "server-side request"],
                lambda value: self._has_any(value, ["callback", "interactsh", "dns hit", "metadata", "169.254.169.254"]),
                "最终结论提到了 SSRF，但缺少回连、metadata 或内网服务响应证据",
            ),
        ]
        missing: list[str] = []
        for claim_category, terms, checker, message in checks:
            if claim_category == category:
                continue
            if self._has_positive_claim(text, terms) and not checker(text):
                missing.append(message)
        return missing

    def _evidence_blob(
        self,
        *,
        finding: dict[str, Any],
        raw: dict[str, Any],
        tool_results: list[dict[str, Any]],
    ) -> str:
        values: list[str] = []

        def collect(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, dict):
                for key, child in value.items():
                    values.append(str(key))
                    collect(child)
                return
            if isinstance(value, list):
                for child in value:
                    collect(child)
                return
            values.append(str(value))

        collect(finding)
        collect(raw)
        collect(tool_results)
        return "\n".join(values).lower()

    def _has_concrete_tool_evidence(self, text: str) -> bool:
        return self._has_any(
            text,
            [
                "tool",
                "curl",
                "http/",
                "status",
                "header",
                "body",
                "response",
                "artifact",
                "stdout",
                "stderr",
                "sqlmap",
                "nuclei",
                "nikto",
                "whatweb",
                "script_runner",
                "bash_runner",
                "web_recon",
                "custom-script-output",
            ],
        )

    def _has_success_status(self, text: str) -> bool:
        if re.search(r"http/\S+\s+2\d\d", text):
            return True
        if re.search(r"\bstatus(?:_code)?[\"'=:\s]+2\d\d\b", text):
            return True
        return self._has_any(text, ["200 ok", "201 created", "202 accepted", "204 no content", "2xx", "status 200"])

    def _has_auth_context(self, text: str) -> bool:
        return self._has_any(
            text,
            [
                "unauthenticated",
                "without authentication",
                "without credentials",
                "anonymous",
                "no auth",
                "no authentication",
                "no www-authenticate",
                "www-authenticate absent",
                "no authentication challenge",
            ],
        )

    def _has_body_data(self, text: str) -> bool:
        return self._has_any(
            text,
            [
                "content-type: application/json",
                "application/json",
                "json keys",
                "response body",
                "body sample",
                "body excerpt",
                "data",
                "email",
                "user",
                "role",
                "product",
                "challenge",
                "token",
                "secret",
                "api_key",
                "password",
            ],
        )

    def _has_sql_injection_proof(self, text: str) -> bool:
        tool_proof = self._has_any(
            text,
            [
                "parameter appears to be injectable",
                "parameter is injectable",
                "sqlmap identified",
                "sqlmap resumed",
                "is vulnerable",
                "boolean-based blind",
                "time-based blind",
                "union query",
                "stacked queries",
                "database management system",
                "current database",
            ],
        )
        db_error = self._has_any(
            text,
            [
                "sql syntax",
                "mysql",
                "mariadb",
                "postgresql",
                "sqlite",
                "ora-",
                "odbc",
                "jdbc",
                "unterminated quoted string",
                "you have an error in your sql syntax",
            ],
        )
        differential = self._has_any(
            text,
            [
                "boolean differential",
                "content differential",
                "time delay",
                "sleep(",
                "benchmark(",
                "waitfor delay",
                "delayed response",
                "true condition",
                "false condition",
            ],
        )
        return tool_proof or db_error or differential

    def _has_positive_claim(self, text: str, terms: list[str]) -> bool:
        negative_markers = [
            "no ",
            "not ",
            "without ",
            "absent ",
            "failed to confirm ",
            "does not ",
            "not enough ",
            "缺少",
            "未发现",
            "不能证明",
        ]
        for term in terms:
            for match in re.finditer(re.escape(term), text):
                before = text[max(0, match.start() - 40) : match.start()]
                if not any(marker in before for marker in negative_markers):
                    return True
        return False

    def _has_any(self, text: str, terms: list[str]) -> bool:
        return any(term.lower() in text for term in terms)

    def _dedupe_text(self, values: list[str]) -> list[str]:
        seen = set()
        result = []
        for value in values:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result[:20]

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
