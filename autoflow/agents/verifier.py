from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from autoflow.agents.base import BaseAgent
from autoflow.agents.tool_loop import AgentToolLoop
from autoflow.flows.models import (
    Finding,
    FindingConfidence,
    FindingSeverity,
    MemoryItem,
    MemoryKind,
)
from autoflow.graph.state import AutoFlowState
from autoflow.settings import settings


VERIFIER_SYSTEM_PROMPT = """You are AutoFlow's VerifierAgent for an authorized security assessment.
You verify tool observations and promote meaningful signals into candidate findings.
You may call memory and observation tools when you need context.
Return only JSON. Do not include markdown.
Do not invent evidence. Findings must be grounded in assets, tool observations, artifacts, or tool results.
Do not perform exploitation, brute force, persistence, destructive writes, evasion, or out-of-scope actions.
"""


@dataclass(frozen=True)
class PromotedSignal:
    category: str
    title: str
    severity: FindingSeverity
    target: str
    description: str
    recommendation: str
    evidence: str
    source: str
    confidence: FindingConfidence
    metadata: dict


class VerifierAgent(BaseAgent):
    """将当前证据汇总为轻量级校验状态。"""

    name = "verifier"

    def __init__(
        self,
        tool_loop: AgentToolLoop | None = None,
        use_tool_calling: bool | None = None,
    ) -> None:
        self.tool_loop = tool_loop
        self.use_tool_calling = use_tool_calling

    async def run(self, state: AutoFlowState) -> AutoFlowState:
        state["current_phase"] = "verification"
        assets = state.get("assets", [])
        executed_tasks = state.get("executed_tasks", [])
        tool_observations = state.get("tool_observations", [])
        open_port_count = sum(len(asset.get("ports", [])) for asset in assets)
        completed_follow_ups = sum(1 for task in executed_tasks if task.get("status") == "completed")
        rule_findings = self._build_findings(assets, executed_tasks, tool_observations)
        findings = self._build_findings_with_tool_loop(state, rule_findings) if self._should_use_tool_loop() else rule_findings
        summary = (
            f"Verified {len(assets)} assets, {open_port_count} open ports, "
            f"{completed_follow_ups} completed follow-up tasks, and {len(findings)} findings"
        )

        state["verification"] = {
            "asset_count": len(assets),
            "open_port_count": open_port_count,
            "completed_follow_up_count": completed_follow_ups,
            "finding_count": len(findings),
            "summary": summary,
        }
        state["findings"] = [finding.model_dump(mode="json") for finding in findings]

        flow = state.get("flow")
        if flow is not None:
            for finding in findings:
                flow.add_finding(finding)
            flow.add_memory(
                MemoryItem(
                    kind=MemoryKind.FINDING if findings else MemoryKind.DECISION,
                    content=summary,
                    source=self.name,
                    references=[finding.id for finding in findings],
                )
            )

        state["next_action"] = "validation"
        return state

    def _should_use_tool_loop(self) -> bool:
        if self.use_tool_calling is not None:
            return self.use_tool_calling
        return bool(settings.llm_api_key)

    def _build_findings_with_tool_loop(
        self,
        state: AutoFlowState,
        rule_findings: list[Finding],
    ) -> list[Finding]:
        loop = self.tool_loop or AgentToolLoop(max_tool_rounds=4, max_tool_calls=6)
        payload = {
            "task": "Review current observations and return candidate findings.",
            "assets": state.get("assets", []),
            "executed_tasks_summary": [
                {
                    "status": item.get("status"),
                    "tool": item.get("task", {}).get("tool"),
                    "profile": item.get("task", {}).get("profile"),
                    "target": item.get("task", {}).get("target"),
                    "summary": item.get("summary", ""),
                    "error": item.get("error", ""),
                }
                for item in state.get("executed_tasks", [])[-40:]
            ],
            "tool_observations": state.get("tool_observations", [])[-40:],
            "rule_candidate_findings": [finding.model_dump(mode="json") for finding in rule_findings],
            "final_output_schema": {
                "findings": [
                    {
                        "title": "short finding title",
                        "status": "candidate",
                        "severity": "info|low|medium|high|critical",
                        "confidence": "low|medium|high",
                        "target": "affected authorized target",
                        "description": "grounded explanation",
                        "evidence": ["specific evidence from observations or tool results"],
                        "recommendation": "next step or remediation guidance",
                        "source": "tool or observation source",
                        "metadata": {"category": "risk category"},
                    }
                ]
            },
        }
        try:
            result = loop.run(
                system_prompt=VERIFIER_SYSTEM_PROMPT,
                user_payload=payload,
                state=state,
                final_repair_instruction="Return a JSON object with a findings array.",
            )
        except Exception:
            return rule_findings

        state["verifier_tool_loop_messages"] = result.messages
        state["verifier_tool_loop_results"] = result.tool_results
        items = result.final.get("findings", [])
        if not isinstance(items, list):
            return rule_findings
        llm_findings = self._coerce_llm_findings(items)
        return self._dedupe_findings([*rule_findings, *llm_findings])

    def _coerce_llm_findings(self, items: list[dict[str, Any]]) -> list[Finding]:
        findings: list[Finding] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            target = self._canonical_target(str(item.get("target") or ""))
            evidence = item.get("evidence") if isinstance(item.get("evidence"), list) else []
            if not title or not target or not evidence:
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            findings.append(
                Finding(
                    title=title,
                    severity=self._severity(str(item.get("severity", "info"))),
                    confidence=self._confidence(str(item.get("confidence", "medium"))),
                    target=target,
                    description=str(item.get("description") or ""),
                    evidence=[str(value) for value in evidence[:10]],
                    recommendation=str(item.get("recommendation") or ""),
                    source=str(item.get("source") or "llm_verifier"),
                    metadata={**metadata, "source": "tool_calling_verifier"},
                )
            )
        return findings

    def _confidence(self, value: str) -> FindingConfidence:
        try:
            return FindingConfidence(value)
        except ValueError:
            return FindingConfidence.MEDIUM

    def _build_findings(
        self,
        assets: list[dict],
        executed_tasks: list[dict],
        tool_observations: list[dict] | None = None,
    ) -> list[Finding]:
        findings = [finding for asset in assets for finding in self._findings_from_asset(asset)]
        findings.extend(self._findings_from_executed_tasks(executed_tasks))
        findings.extend(self._findings_from_tool_observations(tool_observations or []))
        return self._dedupe_findings(findings)

    def _findings_from_asset(self, asset: dict) -> list[Finding]:
        findings: list[Finding] = []
        host = asset.get("ip") or "unknown"
        for port in asset.get("ports", []):
            port_number = port.get("port")
            protocol = port.get("protocol", "tcp")
            service = port.get("service") or "unknown"
            product = port.get("product") or ""
            version = port.get("version") or ""
            target = f"{host}:{port_number}" if port_number else host
            details = " ".join(value for value in [service, product, version] if value)
            findings.append(
                Finding(
                    title=f"Exposed {service} service on {target}",
                    severity=FindingSeverity.INFO,
                    confidence=FindingConfidence.HIGH,
                    target=target,
                    description=f"The target exposes an open {protocol} service: {details}.",
                    evidence=[f"nmap detected {protocol}/{port_number} open as {details}"],
                    recommendation=(
                        "Confirm this service is expected, restricted to the intended network, "
                        "and covered by follow-up validation where appropriate."
                    ),
                    source="nmap",
                    metadata={"asset": asset, "port": port},
                )
            )
        return findings

    def _findings_from_executed_tasks(self, executed_tasks: list[dict]) -> list[Finding]:
        findings: list[Finding] = []
        for item in executed_tasks:
            if item.get("status") != "completed":
                continue
            task = item.get("task", {})
            if task.get("type") != "web_fingerprint":
                continue
            target = task.get("target", "unknown")
            summary = item.get("summary", "")
            title = self._extract_whatweb_title(summary)
            app_name = title or "web application"
            findings.append(
                Finding(
                    title=f"Detected {app_name} on {target}",
                    severity=FindingSeverity.INFO,
                    confidence=FindingConfidence.HIGH if title else FindingConfidence.MEDIUM,
                    target=target,
                    description=(
                        f"Web fingerprinting identified {app_name} on the target. "
                        "This finding is a technology and exposure clue used to choose next tests."
                    ),
                    evidence=[summary],
                    recommendation=(
                        "Use this fingerprint to select safe follow-up checks such as web configuration, "
                        "TLS, and template-based validation."
                    ),
                    source=task.get("tool", "whatweb"),
                    metadata={"executed_task": item},
                )
            )
        return findings

    def _findings_from_tool_observations(self, observations: list[dict]) -> list[Finding]:
        promoted_signals: list[PromotedSignal] = []
        for observation in observations:
            if observation.get("status") != "completed":
                continue
            for signal in observation.get("signals", []):
                promoted = self._promote_signal(observation, signal)
                if promoted is not None:
                    promoted_signals.append(promoted)
        return self._merge_promoted_signals(promoted_signals)

    def _promote_signal(self, observation: dict, signal: dict) -> PromotedSignal | None:
        tool = observation.get("tool", "")
        kind = signal.get("kind", "")
        name = signal.get("name", "tool signal")
        evidence = signal.get("evidence") or observation.get("summary") or observation.get("raw_result", "")[:300]
        target = self._canonical_target(signal.get("target") or observation.get("target", "unknown"))
        category = self._risk_category(tool, kind, name, evidence)
        if category is None:
            return None

        title = self._title_for_category(category, name)
        severity = self._severity_for_category(category, signal.get("severity", "info"))
        return PromotedSignal(
            category=category,
            title=title,
            severity=severity,
            target=target,
            description=self._description_for_category(category, tool, name),
            recommendation=self._recommendation_for_category(category, name),
            evidence=evidence,
            source=tool,
            confidence=FindingConfidence.HIGH if tool in {"nuclei", "script_runner"} else FindingConfidence.MEDIUM,
            metadata={"observation_id": observation.get("id"), "signal": signal, "observation": observation},
        )

    def _risk_category(self, tool: str, kind: str, name: str, evidence: str) -> str | None:
        text = f"{name} {evidence}".lower()
        if kind == "cors_wildcard" or "cors-wildcard" in text or "access-control-allow-origin header: *" in text:
            return "cors_wildcard"
        if "access-control-allow-origin: *" in text or "access-control-allow-origin header: *" in text:
            return "cors_wildcard"
        if kind == "missing_security_header" or "missing-security-headers" in text:
            return f"missing_security_header:{self._header_name_from_text(text)}"
        if "suggested security header missing" in text:
            return f"missing_security_header:{self._header_name_from_text(text)}"
        if "security-config-checks:weak-cache-control" in text:
            return "weak_cache_control"
        if "robots.txt contains" in text or "robots-txt" in text:
            return "robots_txt_exposure"
        if "x-recruiting" in text:
            return "informational_header:x-recruiting"
        if "api-exposure" in text:
            return "api_exposure"
        if "debug-endpoints" in text:
            return "debug_endpoint_exposed"
        if "directory-listing" in text or "directory listing" in text:
            return "directory_listing"
        if "public-config-files" in text:
            return "public_config_exposure"
        if "sensitive-paths" in text:
            return "sensitive_path_exposed"
        if "tech-stack-detection" in text:
            return "tech_stack_fingerprint"
        if any(keyword in text for keyword in ["directory indexing", "backup", "vulnerability", "cve", "osvdb"]):
            return "web_risk_observation"
        return None

    def _header_name_from_text(self, text: str) -> str:
        aliases = {
            "missing-csp": "content-security-policy",
            " csp": "content-security-policy",
            "missing-referrer-policy": "referrer-policy",
            "missing-permissions-policy": "permissions-policy",
            "missing-hsts": "strict-transport-security",
        }
        for alias, header in aliases.items():
            if alias in text:
                return header
        known_headers = [
            "content-security-policy",
            "referrer-policy",
            "permissions-policy",
            "strict-transport-security",
            "x-frame-options",
            "x-content-type-options",
        ]
        for header in known_headers:
            if header in text:
                return header
        match = re.search(r"missing[:\s-]+([a-z0-9-]+)", text)
        return match.group(1) if match else "unknown"

    def _title_for_category(self, category: str, name: str) -> str:
        if category == "cors_wildcard":
            return "Wildcard CORS header"
        if category.startswith("missing_security_header:"):
            return f"Missing security header: {category.split(':', 1)[1]}"
        if category == "robots_txt_exposure":
            return "Robots.txt exposes crawl hints"
        if category == "informational_header:x-recruiting":
            return "Informational x-recruiting header exposed"
        if category == "weak_cache_control":
            return "Weak cache-control policy"
        if category == "api_exposure":
            return "Exposed API endpoint"
        if category == "debug_endpoint_exposed":
            return "Debug or diagnostic endpoint exposed"
        if category == "directory_listing":
            return "Directory listing exposed"
        if category == "public_config_exposure":
            return "Public configuration file exposed"
        if category == "sensitive_path_exposed":
            return "Sensitive or interesting path exposed"
        if category == "tech_stack_fingerprint":
            return "Technology stack fingerprint"
        return f"Web risk observation: {name}"

    def _description_for_category(self, category: str, tool: str, name: str) -> str:
        if category == "cors_wildcard":
            return f"{tool} observed a wildcard Access-Control-Allow-Origin response."
        if category.startswith("missing_security_header:"):
            return f"{tool} observed that a common browser-facing security header is missing."
        if category == "robots_txt_exposure":
            return f"{tool} observed robots.txt entries that may reveal application paths."
        if category == "informational_header:x-recruiting":
            return f"{tool} observed an x-recruiting header that discloses an application route or hint."
        if category == "weak_cache_control":
            return f"{tool} observed a response without strong no-store/no-cache cache-control directives."
        if category == "api_exposure":
            return f"{tool} observed a reachable API endpoint that may need authorization or data exposure validation."
        if category == "debug_endpoint_exposed":
            return f"{tool} observed a debug or diagnostic endpoint that may expose runtime details."
        if category == "directory_listing":
            return f"{tool} observed a directory listing style response."
        if category == "public_config_exposure":
            return f"{tool} observed a publicly reachable configuration or metadata file."
        if category == "sensitive_path_exposed":
            return f"{tool} observed a sensitive or interesting path that should be classified."
        if category == "tech_stack_fingerprint":
            return f"{tool} observed technology stack indicators useful for selecting validation checks."
        return f"{tool} observed {name} during automated validation."

    def _recommendation_for_category(self, category: str, name: str) -> str:
        if category == "cors_wildcard":
            return "Restrict Access-Control-Allow-Origin to trusted origins and avoid credentials with wildcard origins."
        if category.startswith("missing_security_header:"):
            header = category.split(":", 1)[1]
            return f"Add and tune the {header} response header where appropriate."
        if category == "robots_txt_exposure":
            return "Review robots.txt entries and ensure they do not reveal sensitive or unintended paths."
        if category == "informational_header:x-recruiting":
            return "Confirm the disclosed route or hint is intentional and does not reveal sensitive workflow details."
        if category == "weak_cache_control":
            return "Apply appropriate Cache-Control directives on sensitive or dynamic responses."
        if category == "api_exposure":
            return "Verify authorization, data sensitivity, and intended exposure for this API endpoint."
        if category == "debug_endpoint_exposed":
            return "Restrict debug and diagnostic endpoints to trusted operators or remove them from production."
        if category == "directory_listing":
            return "Disable directory listing or ensure listed files are intentionally public."
        if category == "public_config_exposure":
            return "Remove public access to sensitive configuration files and review exposed metadata."
        if category == "sensitive_path_exposed":
            return "Classify the path and verify whether it exposes sensitive functionality or data."
        if category == "tech_stack_fingerprint":
            return "Use the fingerprint to select framework-specific checks; this is not a vulnerability by itself."
        return "Review this observation and decide whether follow-up validation is needed."

    def _severity_for_category(self, category: str, raw_value: str) -> FindingSeverity:
        if category.startswith("missing_security_header:"):
            return FindingSeverity.LOW
        if category == "cors_wildcard":
            return FindingSeverity.LOW
        if category in {
            "api_exposure",
            "debug_endpoint_exposed",
            "directory_listing",
            "public_config_exposure",
        }:
            return self._severity(raw_value)
        if category == "weak_cache_control":
            return FindingSeverity.LOW
        if category in {"sensitive_path_exposed", "tech_stack_fingerprint"}:
            return FindingSeverity.INFO if category == "tech_stack_fingerprint" else FindingSeverity.LOW
        if category in {"robots_txt_exposure", "informational_header:x-recruiting"}:
            return FindingSeverity.INFO
        return self._severity(raw_value)

    def _merge_promoted_signals(self, signals: list[PromotedSignal]) -> list[Finding]:
        grouped: dict[tuple[str, str], list[PromotedSignal]] = {}
        for signal in signals:
            grouped.setdefault((signal.category, signal.target), []).append(signal)

        findings: list[Finding] = []
        for (_, _), items in grouped.items():
            first = items[0]
            sources = sorted({item.source for item in items if item.source})
            evidence: list[str] = []
            metadata_signals = []
            for item in items:
                if item.evidence and item.evidence not in evidence:
                    evidence.append(item.evidence)
                metadata_signals.append(item.metadata)
            severity = max((item.severity for item in items), key=self._severity_rank)
            confidence = FindingConfidence.HIGH if len(sources) > 1 or first.confidence == FindingConfidence.HIGH else first.confidence
            findings.append(
                Finding(
                    title=first.title,
                    severity=severity,
                    confidence=confidence,
                    target=first.target,
                    description=first.description,
                    evidence=evidence[:10],
                    recommendation=first.recommendation,
                    source=",".join(sources) if sources else first.source,
                    metadata={"category": first.category, "signals": metadata_signals},
                )
            )
        return findings

    def _severity_rank(self, severity: FindingSeverity) -> int:
        ranks = {
            FindingSeverity.INFO: 0,
            FindingSeverity.LOW: 1,
            FindingSeverity.MEDIUM: 2,
            FindingSeverity.HIGH: 3,
            FindingSeverity.CRITICAL: 4,
        }
        return ranks[severity]

    def _severity(self, value: str) -> FindingSeverity:
        try:
            return FindingSeverity(value)
        except ValueError:
            return FindingSeverity.INFO

    def _dedupe_findings(self, findings: list[Finding]) -> list[Finding]:
        seen: set[tuple[str, str, str]] = set()
        unique: list[Finding] = []
        for finding in findings:
            finding.target = self._canonical_target(finding.target)
            key = (finding.title, finding.target, finding.source)
            if key in seen:
                continue
            seen.add(key)
            unique.append(finding)
        return unique

    def _canonical_target(self, target: str) -> str:
        if not target:
            return target
        parsed = urlsplit(target)
        if not parsed.scheme or not parsed.netloc:
            return target
        path = "" if parsed.path == "/" else parsed.path.rstrip("/.,;)")
        if parsed.fragment and not path:
            path = "/"
        return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))

    def _extract_whatweb_title(self, summary: str) -> str:
        match = re.search(r"Title\[([^\]]+)\]", summary)
        return match.group(1).strip() if match else ""
