from __future__ import annotations

from autoflow.agents.base import BaseAgent
from autoflow.agents.tool_loop import AgentToolLoop
from autoflow.flows.models import (
    FindingStatus,
    MemoryItem,
    MemoryKind,
    RiskLevel,
    TestPlanAction,
    ValidationPlan,
)
from autoflow.graph.state import AutoFlowState
from autoflow.memory.agent_memory import AgentMemoryBuilder
from autoflow.runtime.actions import canonical_target
from autoflow.settings import settings
from autoflow.tools.manifest import ToolManifestRegistry


VALIDATION_SYSTEM_PROMPT = """You are AutoFlow's ValidationAgent for an authorized lab security assessment.
You turn candidate findings into concrete validation plans.
You may call memory and observation tools when you need context, but final output must be JSON.
Current lab mode allows read-only and bounded validation actions without approval.
You may create generated script actions only when existing tool profiles are insufficient.
Generated scripts must have a precise script_goal, stay scoped to the target, be bounded, and collect evidence only.
You may create container lab shell actions for command-line validation inside the disposable tool container.
Shell actions must use $TARGET or the exact authorized target and should save useful evidence to stdout or $ARTIFACT_DIR.
Do not invent out-of-scope targets. Do not create destructive, persistence, evasion, brute force, or lateral movement actions.
"""


class ValidationAgent(BaseAgent):
    """根据 Candidate Finding 生成后续漏洞验证策略。"""

    name = "validation"

    def __init__(
        self,
        require_approval: bool = False,
        tool_loop: AgentToolLoop | None = None,
        use_tool_calling: bool | None = None,
        tool_manifest: ToolManifestRegistry | None = None,
        memory_builder: AgentMemoryBuilder | None = None,
    ) -> None:
        self.require_approval = require_approval
        self.tool_loop = tool_loop
        self.use_tool_calling = use_tool_calling
        self.tool_manifest = tool_manifest or ToolManifestRegistry()
        self.memory_builder = memory_builder or AgentMemoryBuilder()

    async def run(self, state: AutoFlowState) -> AutoFlowState:
        state["current_phase"] = "validation_planning"
        findings = state.get("findings", [])
        existing = state.get("validation_plans", [])
        rule_plans = self._build_validation_plans(findings, existing)
        plans = (
            self._build_validation_plans_with_tool_loop(state, findings, existing, rule_plans)
            if self._should_use_tool_loop()
            else rule_plans
        )
        state["validation_plans"] = [*existing, *[plan.model_dump(mode="json") for plan in plans]]

        flow = state.get("flow")
        if flow is not None:
            for plan in plans:
                flow.add_validation_plan(plan)
            flow.add_memory(
                MemoryItem(
                    kind=MemoryKind.DECISION,
                    content=f"ValidationAgent produced {len(plans)} validation plans",
                    source=self.name,
                    references=[plan.finding_id for plan in plans],
                )
            )

        auto_execute = state.get("rules_of_engagement", {}).get("validation_auto_execute", True)
        state["next_action"] = "validation_execute" if auto_execute and self._has_executable_validation_actions(plans) else "strategy"
        return state

    def _has_executable_validation_actions(self, plans: list[ValidationPlan]) -> bool:
        return any(plan.actions for plan in plans)

    def _should_use_tool_loop(self) -> bool:
        if self.use_tool_calling is not None:
            return self.use_tool_calling
        return bool(settings.llm_api_key)

    def _build_validation_plans_with_tool_loop(
        self,
        state: AutoFlowState,
        findings: list[dict],
        existing: list[dict],
        rule_plans: list[ValidationPlan],
    ) -> list[ValidationPlan]:
        loop = self.tool_loop or AgentToolLoop(max_tool_rounds=4, max_tool_calls=6)
        payload = {
            "task": "Create concrete validation plans for candidate findings.",
            "lab_mode": True,
            "memory_pack": self.memory_builder.build(state, persisted_memory=state.get("agent_memory")),
            "findings": findings[-40:],
            "existing_validation_plans": existing[-40:],
            "rule_candidate_validation_plans": [plan.model_dump(mode="json") for plan in rule_plans],
            "tool_observations": state.get("tool_observations", [])[-40:],
            "known_targets": state.get("target_scope", []),
            "allowed_action_kinds": ["web_recon", "script", "tool", "shell"],
            "tool_manifest": self.tool_manifest.prompt_manifest({"validation", "artifact_audit", "discovery"}),
            "preferred_tools": self._preferred_validation_tools(),
            "validation_strategy_guidance": [
                "API exposure: collect status, headers, content type, JSON keys, unauthenticated body sample, and sensitivity hints.",
                "Directory listing: collect href entries and highlight backups, config, logs, keys, database files, and archives.",
                "Debug endpoints: collect runtime keyword matches such as env, stack, process, heap, trace, metrics, token, secret.",
                "Public config files: collect status, content type, dependency/version hints, endpoints, and secret-like tokens.",
                "Security headers and CORS: confirm the exact headers on representative responses and quote the raw evidence.",
                "Use shell actions for compact curl/grep/jq evidence and script_runner templates for structured summaries.",
            ],
            "final_output_schema": {
                "validation_plans": [
                    {
                        "finding_id": "existing finding id",
                        "target": "authorized target",
                        "objective": "what to verify",
                        "risk_level": "low|medium|high|critical",
                        "requires_approval": False,
                        "actions": [
                            {
                                "name": "action name",
                                "action_kind": "web_recon|script|tool|shell",
                                "tool": "web_recon|script_runner|bash_runner|curl|nuclei|whatweb|nikto",
                                "profile": "tool profile",
                                "target": "authorized target",
                                "risk_level": "low|medium",
                                "requires_approval": False,
                                "args": {},
                                "shell_command": "optional container-only bash command using $TARGET",
                                "shell_policy_profile": "container_lab_shell|low_readonly_http|medium_artifact_shell",
                                "script_template": "optional script template",
                                "script_goal": "optional precise generated-script goal when script_template is not used",
                                "script_policy_profile": "low_readonly_http|medium_artifact_script|high_lab_poc",
                                "max_script_attempts": 2,
                                "rationale": "why this action helps verify the finding",
                            }
                        ],
                        "success_criteria": ["condition indicating validated"],
                        "failure_criteria": ["condition indicating false positive or insufficient evidence"],
                        "rationale": "grounded reason",
                        "metadata": {"category": "finding category"},
                    }
                ]
            },
        }
        try:
            result = loop.run(
                system_prompt=VALIDATION_SYSTEM_PROMPT,
                user_payload=payload,
                state=state,
                final_repair_instruction="Return a JSON object with a validation_plans array.",
            )
        except Exception:
            return rule_plans

        state["validation_tool_loop_messages"] = result.messages
        state["validation_tool_loop_results"] = result.tool_results
        items = result.final.get("validation_plans", [])
        if not isinstance(items, list):
            return rule_plans
        llm_plans = self._coerce_llm_validation_plans(items, findings)
        return self._dedupe_validation_plan_models([*rule_plans, *llm_plans])

    def _preferred_validation_tools(self) -> list[str]:
        tools = []
        for item in self.tool_manifest.prompt_manifest({"validation", "artifact_audit", "discovery"}):
            tool = item.get("tool")
            profile = item.get("profile")
            if tool and profile:
                tools.append(f"{tool}/{profile}: {item.get('purpose', '')}")
        tools.extend(
            [
                "bash_runner/bounded_bash: container-only lab shell commands using metadata.shell_command",
                "script_runner/generated_python_script: use metadata.script_goal when no existing profile fits",
            ]
        )
        return tools

    def _coerce_llm_validation_plans(
        self,
        items: list[dict],
        findings: list[dict],
    ) -> list[ValidationPlan]:
        finding_ids = {finding.get("id") for finding in findings if finding.get("id")}
        plans: list[ValidationPlan] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            finding_id = str(item.get("finding_id") or "")
            target = canonical_target(str(item.get("target") or ""))
            objective = str(item.get("objective") or "")
            if not finding_id or finding_id not in finding_ids or not target or not objective:
                continue
            actions = self._coerce_llm_actions(item.get("actions", []), target)
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            plans.append(
                ValidationPlan(
                    finding_id=finding_id,
                    target=target,
                    objective=objective,
                    risk_level=self._risk(str(item.get("risk_level", "medium"))),
                    requires_approval=bool(item.get("requires_approval", self.require_approval)),
                    actions=actions,
                    success_criteria=[str(value) for value in item.get("success_criteria", [])[:10]]
                    if isinstance(item.get("success_criteria"), list)
                    else [],
                    failure_criteria=[str(value) for value in item.get("failure_criteria", [])[:10]]
                    if isinstance(item.get("failure_criteria"), list)
                    else [],
                    rationale=str(item.get("rationale") or ""),
                    metadata={**metadata, "source": "tool_calling_validation"},
                )
            )
        return plans

    def _coerce_llm_actions(self, items, default_target: str) -> list[TestPlanAction]:
        if not isinstance(items, list):
            return []
        actions: list[TestPlanAction] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or "")
            profile = str(item.get("profile") or "")
            action_kind = str(item.get("action_kind") or "tool")
            target = canonical_target(str(item.get("target") or default_target))
            if not tool or not profile or not target:
                continue
            risk = self._risk(str(item.get("risk_level", "low")))
            script_template = item.get("script_template")
            script_goal = str(item.get("script_goal") or "")
            shell_command = str(item.get("shell_command") or "")
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            if tool == "script_runner" and not script_template and not script_goal and profile != "generated_python_script":
                script_template = profile
            action_metadata = {**metadata, "source": "tool_calling_validation"}
            if action_kind == "shell" or tool == "bash_runner":
                tool = "bash_runner"
                profile = profile or "bounded_bash"
                action_kind = "shell"
                if shell_command:
                    action_metadata["shell_command"] = shell_command
                action_metadata["shell_policy_profile"] = str(
                    item.get("shell_policy_profile") or metadata.get("shell_policy_profile") or "container_lab_shell"
                )
            if script_goal:
                action_metadata["script_goal"] = script_goal
                action_metadata["script_policy_profile"] = str(
                    item.get("script_policy_profile") or metadata.get("script_policy_profile") or "medium_artifact_script"
                )
                action_metadata["max_script_attempts"] = str(
                    item.get("max_script_attempts") or metadata.get("max_script_attempts") or 2
                )
            actions.append(
                TestPlanAction(
                    name=str(item.get("name") or f"Run {tool}/{profile}"),
                    action_kind=action_kind,
                    tool=tool,
                    profile=profile,
                    target=target,
                    risk_level=risk,
                    requires_approval=bool(item.get("requires_approval", self.require_approval)),
                    expected_impact="Bounded validation action in authorized lab mode.",
                    rationale=str(item.get("rationale") or ""),
                    args={str(key): str(value) for key, value in (item.get("args") or {}).items()}
                    if isinstance(item.get("args"), dict)
                    else {},
                    script_template=str(script_template) if script_template else None,
                    metadata=action_metadata,
                )
            )
        return actions

    def _dedupe_validation_plan_models(self, plans: list[ValidationPlan]) -> list[ValidationPlan]:
        seen: set[tuple[str, str, str]] = set()
        result: list[ValidationPlan] = []
        for plan in plans:
            key = (plan.finding_id, canonical_target(plan.target), plan.objective)
            if key in seen:
                continue
            seen.add(key)
            result.append(plan)
        return result

    def _risk(self, value: str) -> RiskLevel:
        try:
            return RiskLevel(value)
        except ValueError:
            return RiskLevel.LOW

    def _build_validation_plans(self, findings: list[dict], existing: list[dict]) -> list[ValidationPlan]:
        seen = {self._existing_plan_key(plan) for plan in existing}
        plans: list[ValidationPlan] = []
        for finding in findings:
            if finding.get("status", FindingStatus.CANDIDATE.value) != FindingStatus.CANDIDATE.value:
                continue
            target = canonical_target(finding.get("target", ""))
            key = self._finding_key(finding, target)
            if key in seen:
                continue
            plan = self._plan_for_finding(finding, target)
            if plan is not None:
                plans.append(plan)
                seen.add(key)
        return plans

    def _existing_plan_key(self, plan: dict) -> tuple[str, str, str]:
        metadata = plan.get("metadata") or {}
        finding = metadata.get("finding") if isinstance(metadata.get("finding"), dict) else {}
        target = canonical_target(plan.get("target", ""))
        if finding:
            return self._finding_key(finding, target)
        return ("finding_id", str(plan.get("finding_id", "")), target)

    def _finding_key(self, finding: dict, target: str) -> tuple[str, str, str]:
        category = self._category(finding) or "uncategorized"
        title = str(finding.get("title") or "")
        if finding.get("id") and category == "uncategorized" and not title:
            return ("finding_id", str(finding.get("id")), target)
        return (category, title, target)

    def _plan_for_finding(self, finding: dict, target: str) -> ValidationPlan | None:
        category = self._category(finding)
        if category == "cors_wildcard":
            return self._cors_plan(finding, target)
        if category.startswith("missing_security_header:"):
            return self._security_header_plan(finding, target)
        if category == "weak_cache_control":
            return self._cache_control_plan(finding, target)
        if category == "api_exposure":
            return self._api_exposure_plan(finding, target)
        if category == "debug_endpoint_exposed":
            return self._debug_endpoint_plan(finding, target)
        if category == "directory_listing":
            return self._directory_listing_plan(finding, target)
        if category == "public_config_exposure":
            return self._public_config_plan(finding, target)
        if category in {"sensitive_path_exposed", "robots_txt_exposure", "informational_header:x-recruiting"}:
            return self._path_followup_plan(finding, target)
        if category == "tech_stack_fingerprint":
            return self._tech_stack_plan(finding, target)
        if category == "web_risk_observation":
            return self._generic_web_risk_plan(finding, target)
        return None

    def _cors_plan(self, finding: dict, target: str) -> ValidationPlan:
        return ValidationPlan(
            finding_id=finding.get("id", ""),
            target=target,
            objective="Verify whether wildcard CORS is exploitable in the current application context.",
            risk_level=RiskLevel.MEDIUM,
            requires_approval=self.require_approval,
            actions=[
                self._shell_validation_action(
                    target,
                    finding,
                    "Reproduce wildcard CORS header with controlled Origin",
                    'printf "[cors headers]\\n"; '
                    'curl -sS -i -L --max-time 10 -A AutoFlow-validation -H "Origin: null" "$TARGET" '
                    '| tee "$ARTIFACT_DIR/cors-response.txt" '
                    '| grep -iE "^(HTTP/|access-control-allow-origin|access-control-allow-credentials|vary):" || true',
                    RiskLevel.LOW,
                    "Collect raw CORS response headers with a controlled Origin value.",
                ),
                self._header_check_action(target, finding),
                self._script_probe_action(
                    target,
                    finding,
                    "Compare CORS behavior with controlled Origin headers",
                    "cors_probe",
                    RiskLevel.MEDIUM,
                    "Send read-only requests with controlled Origin values and compare CORS response headers.",
                ),
            ],
            success_criteria=[
                "Response reflects permissive Access-Control-Allow-Origin behavior.",
                "Sensitive or authenticated endpoints also expose permissive CORS headers.",
            ],
            failure_criteria=[
                "CORS wildcard only appears on static unauthenticated content.",
                "No credentials or sensitive endpoints are reachable under permissive CORS.",
            ],
            rationale="Wildcard CORS is a candidate risk; exploitability depends on credentials, sensitive endpoints, and browser behavior.",
            metadata={"finding": finding, "category": "cors_wildcard"},
        )

    def _security_header_plan(self, finding: dict, target: str) -> ValidationPlan:
        header = self._category(finding).split(":", 1)[1]
        return ValidationPlan(
            finding_id=finding.get("id", ""),
            target=target,
            objective=f"Confirm missing {header} and decide whether it materially affects the application.",
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            actions=[
                self._shell_validation_action(
                    target,
                    finding,
                    f"Reproduce missing {header} with curl headers",
                    f'printf "[security headers]\\n"; '
                    f'curl -sS -I -L --max-time 10 -A AutoFlow-validation "$TARGET" '
                    f'| tee "$ARTIFACT_DIR/security-headers.txt"; '
                    f'if ! grep -iq "^{header}:" "$ARTIFACT_DIR/security-headers.txt"; '
                    f'then echo "missing: {header}"; else echo "present: {header}"; fi',
                    RiskLevel.LOW,
                    f"Collect raw headers and explicitly mark whether {header} is missing.",
                ),
                self._header_check_action(target, finding),
            ],
            success_criteria=[f"The {header} header is absent on representative application responses."],
            failure_criteria=[f"The {header} header is present or intentionally omitted only on irrelevant static content."],
            rationale="Header checks are read-only and can be safely confirmed before deeper validation.",
            metadata={"finding": finding, "category": self._category(finding), "header": header},
        )

    def _cache_control_plan(self, finding: dict, target: str) -> ValidationPlan:
        return ValidationPlan(
            finding_id=finding.get("id", ""),
            target=target,
            objective="Confirm whether cache-control policy is weak on sensitive pages.",
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            actions=[
                self._shell_validation_action(
                    target,
                    finding,
                    "Reproduce weak cache-control headers",
                    'printf "[cache headers]\\n"; '
                    'curl -sS -I -L --max-time 10 -A AutoFlow-validation "$TARGET" '
                    '| tee "$ARTIFACT_DIR/cache-headers.txt"; '
                    'printf "\\n[cache-control]\\n"; '
                    'grep -iE "^(cache-control|pragma|expires):" "$ARTIFACT_DIR/cache-headers.txt" || echo "missing cache headers"',
                    RiskLevel.LOW,
                    "Collect cache-related headers from the candidate response.",
                ),
                self._header_check_action(target, finding),
            ],
            success_criteria=["Sensitive or dynamic responses lack no-store/no-cache style cache-control protections."],
            failure_criteria=["Only static public assets have cacheable responses."],
            rationale="Cache-control weakness needs context; first confirm where the header is missing.",
            metadata={"finding": finding, "category": "weak_cache_control"},
        )

    def _api_exposure_plan(self, finding: dict, target: str) -> ValidationPlan:
        return ValidationPlan(
            finding_id=finding.get("id", ""),
            target=target,
            objective="Verify whether the exposed API returns sensitive data or lacks expected authorization.",
            risk_level=RiskLevel.MEDIUM,
            requires_approval=self.require_approval,
            actions=[
                self._web_recon_action(target, finding, "Fetch exposed API endpoint"),
                self._curl_action(target, finding, "Fetch API response with curl", "get_with_headers", RiskLevel.LOW),
                self._shell_validation_action(
                    target,
                    finding,
                    "Reproduce API exposure and summarize JSON keys",
                    'printf "[api headers]\\n"; '
                    'curl -sS -i -L --max-time 10 -A AutoFlow-validation "$TARGET" '
                    '| tee "$ARTIFACT_DIR/api-response.txt" '
                    '| sed -n "1,80p"; '
                    'printf "\\n[api status/content]\\n"; '
                    'grep -iE "^(HTTP/|content-type|www-authenticate|cache-control):" "$ARTIFACT_DIR/api-response.txt" || true; '
                    'printf "\\n[api json keys]\\n"; '
                    'sed -n "/^\\r\\{0,1\\}$/,$p" "$ARTIFACT_DIR/api-response.txt" '
                    '| jq -r \'if type=="object" then keys[] elif type=="array" and length>0 and (.[0]|type=="object") then .[0] | keys[] else empty end\' 2>/dev/null '
                    '| head -n 80 || true; '
                    'printf "\\n[sensitivity_hints]\\n"; '
                    'grep -Eio "password|token|secret|email|role|user|admin|api[_-]?key" "$ARTIFACT_DIR/api-response.txt" '
                    '| sort -u | head -n 40 || true',
                    RiskLevel.MEDIUM,
                    "Capture unauthenticated API response evidence, JSON shape, and sensitive-field hints.",
                ),
                self._script_probe_action(
                    target,
                    finding,
                    "Compare unauthenticated and authenticated API behavior",
                    "api_endpoint_probe",
                    RiskLevel.MEDIUM,
                    "Fetch the API without credentials and summarize status, content type, JSON keys, and sensitivity hints.",
                ),
            ],
            success_criteria=[
                "Unauthenticated requests return sensitive fields or business data.",
                "Endpoint behavior differs in a way that suggests missing authorization checks.",
            ],
            failure_criteria=[
                "Endpoint returns only public metadata.",
                "Endpoint correctly returns 401/403 or anonymous-safe data.",
            ],
            rationale="API exposure is a candidate issue; validation must check authorization and data sensitivity.",
            metadata={"finding": finding, "category": "api_exposure"},
        )

    def _debug_endpoint_plan(self, finding: dict, target: str) -> ValidationPlan:
        return ValidationPlan(
            finding_id=finding.get("id", ""),
            target=target,
            objective="Verify whether the debug or diagnostic endpoint exposes sensitive runtime information.",
            risk_level=RiskLevel.MEDIUM,
            requires_approval=self.require_approval,
            actions=[
                self._web_recon_action(target, finding, "Fetch debug endpoint safely"),
                self._curl_action(target, finding, "Fetch debug endpoint headers and body", "get_with_headers", RiskLevel.LOW),
                self._shell_validation_action(
                    target,
                    finding,
                    "Reproduce debug endpoint information exposure",
                    'printf "[debug response]\\n"; '
                    'curl -sS -i -L --max-time 10 -A AutoFlow-validation "$TARGET" '
                    '| tee "$ARTIFACT_DIR/debug-response.txt" '
                    '| sed -n "1,100p"; '
                    'printf "\\n[matched_keywords]\\n"; '
                    'grep -Eio "process|heap|memory|uptime|nodejs|express|stack|trace|exception|env|secret|token|password|prometheus|metrics" '
                    '"$ARTIFACT_DIR/debug-response.txt" | sort -u | head -n 60 || true',
                    RiskLevel.MEDIUM,
                    "Collect raw debug endpoint output and highlight runtime or secret-like keywords.",
                ),
                self._script_probe_action(
                    target,
                    finding,
                    "Review debug endpoint content for secrets and internals",
                    "debug_endpoint_probe",
                    RiskLevel.MEDIUM,
                    "Inspect read-only response content for metrics, internal paths, stack traces, and configuration hints.",
                ),
            ],
            success_criteria=["Debug endpoint exposes runtime internals, metrics, environment data, or stack traces."],
            failure_criteria=["Endpoint is protected, unavailable, or only exposes harmless health status."],
            rationale="Debug endpoints can be sensitive, but confirmation requires content review.",
            metadata={"finding": finding, "category": "debug_endpoint_exposed"},
        )

    def _directory_listing_plan(self, finding: dict, target: str) -> ValidationPlan:
        return ValidationPlan(
            finding_id=finding.get("id", ""),
            target=target,
            objective="Verify whether directory listing exposes sensitive files.",
            risk_level=RiskLevel.MEDIUM,
            requires_approval=self.require_approval,
            actions=[
                self._web_recon_action(target, finding, "Fetch directory listing"),
                self._curl_action(target, finding, "Fetch directory listing with curl", "get", RiskLevel.LOW),
                self._shell_validation_action(
                    target,
                    finding,
                    "Reproduce directory listing and highlight interesting entries",
                    'printf "[directory listing]\\n"; '
                    'curl -sS -L --max-time 10 -A AutoFlow-validation "$TARGET" '
                    '| tee "$ARTIFACT_DIR/directory-listing.html" '
                    '| sed -n "1,100p";\n'
                    'python3 - "$ARTIFACT_DIR/directory-listing.html" <<\'PY\'\n'
                    'import re, sys\n'
                    'path = sys.argv[1]\n'
                    'text = open(path, encoding="utf-8", errors="replace").read()\n'
                    'entries = re.findall(r\'href=["\\\']([^"\\\']+)["\\\']\', text, flags=re.I)\n'
                    'patterns = [".bak", ".backup", ".old", ".zip", ".tar", ".gz", ".db", ".sqlite", ".key", ".pem", ".log", ".env", "config", "secret", "password"]\n'
                    'interesting = [item for item in entries if any(pattern in item.lower() for pattern in patterns)]\n'
                    'print("\\n[entries]")\n'
                    'print("\\n".join(entries[:120]))\n'
                    'print("\\n[interesting_entries]")\n'
                    'print("\\n".join(interesting[:80]))\n'
                    'PY',
                    RiskLevel.MEDIUM,
                    "Collect listing entries and highlight files that can increase impact.",
                ),
                self._script_probe_action(
                    target,
                    finding,
                    "Classify listed files and select safe downloads",
                    "directory_listing_probe",
                    RiskLevel.MEDIUM,
                    "Parse listed filenames and classify whether the listing exposes backup, config, key, log, or database files.",
                ),
            ],
            success_criteria=["Directory listing exposes backup, config, credential, log, or source files."],
            failure_criteria=["Listing contains only public static assets."],
            rationale="Directory listing is a strong candidate signal; impact depends on listed file content.",
            metadata={"finding": finding, "category": "directory_listing"},
        )

    def _public_config_plan(self, finding: dict, target: str) -> ValidationPlan:
        return ValidationPlan(
            finding_id=finding.get("id", ""),
            target=target,
            objective="Verify whether the public file contains sensitive configuration or dependency intelligence.",
            risk_level=RiskLevel.MEDIUM,
            requires_approval=self.require_approval,
            actions=[
                self._web_recon_action(target, finding, "Fetch public configuration file"),
                self._curl_action(target, finding, "Fetch public file with curl", "get_with_headers", RiskLevel.LOW),
                self._shell_validation_action(
                    target,
                    finding,
                    "Reproduce public config exposure and search sensitive hints",
                    'printf "[public config response]\\n"; '
                    'curl -sS -i -L --max-time 10 -A AutoFlow-validation "$TARGET" '
                    '| tee "$ARTIFACT_DIR/public-config.txt" '
                    '| sed -n "1,120p"; '
                    'printf "\\n[sensitivity_hints]\\n"; '
                    'grep -Eio "password|secret|token|api[_-]?key|private|database|mongodb|redis|jwt|credential|client[_-]?secret" '
                    '"$ARTIFACT_DIR/public-config.txt" | sort -u | head -n 80 || true; '
                    'printf "\\n[dependency/version hints]\\n"; '
                    'grep -Eio "\\"(version|dependencies|devDependencies|node|express|angular)\\"" "$ARTIFACT_DIR/public-config.txt" '
                    '| sort -u | head -n 80 || true',
                    RiskLevel.MEDIUM,
                    "Collect public file content and highlight secrets, endpoints, and version/dependency clues.",
                ),
                self._script_probe_action(
                    target,
                    finding,
                    "Inspect public config for secrets or vulnerable dependencies",
                    "public_config_probe",
                    RiskLevel.MEDIUM,
                    "Fetch the public file and classify secrets, endpoints, dependency versions, and deployment metadata.",
                ),
            ],
            success_criteria=["File contains secrets, internal endpoints, dependency versions, or deployment metadata."],
            failure_criteria=["File contains only intended public metadata."],
            rationale="Public config files range from harmless metadata to high-impact secret exposure.",
            metadata={"finding": finding, "category": "public_config_exposure"},
        )

    def _path_followup_plan(self, finding: dict, target: str) -> ValidationPlan:
        return ValidationPlan(
            finding_id=finding.get("id", ""),
            target=target,
            objective="Fetch and classify the discovered path before selecting deeper tests.",
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            actions=[
                self._web_recon_action(target, finding, "Refresh recon for discovered path"),
                self._shell_validation_action(
                    target,
                    finding,
                    "Fetch discovered path and classify response",
                    'printf "[path response]\\n"; '
                    'curl -sS -i -L --max-time 10 -A AutoFlow-validation "$TARGET" '
                    '| tee "$ARTIFACT_DIR/path-response.txt" '
                    '| sed -n "1,100p"; '
                    'printf "\\n[path signals]\\n"; '
                    'grep -Eio "login|admin|debug|swagger|graphql|api|token|secret|password|directory listing|index of" '
                    '"$ARTIFACT_DIR/path-response.txt" | sort -u | head -n 80 || true',
                    RiskLevel.LOW,
                    "Collect raw response and classify whether the path is actionable.",
                ),
            ],
            success_criteria=["Path returns meaningful content, forms, APIs, files, or links for follow-up."],
            failure_criteria=["Path is unavailable, redirects to generic content, or has no useful signal."],
            rationale="Discovered paths should first be parsed into context before validation or exploitation.",
            metadata={"finding": finding, "category": self._category(finding)},
        )

    def _tech_stack_plan(self, finding: dict, target: str) -> ValidationPlan:
        return ValidationPlan(
            finding_id=finding.get("id", ""),
            target=target,
            objective="Use technology fingerprints to select later framework-specific checks.",
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            actions=[],
            success_criteria=["Technology fingerprint is corroborated by multiple observations or static assets."],
            failure_criteria=["Fingerprint is generic or not actionable."],
            rationale="Technology fingerprints are context for strategy, not vulnerabilities by themselves.",
            metadata={"finding": finding, "category": "tech_stack_fingerprint"},
        )

    def _generic_web_risk_plan(self, finding: dict, target: str) -> ValidationPlan:
        return ValidationPlan(
            finding_id=finding.get("id", ""),
            target=target,
            objective="Clarify the observed web risk and determine a safe validation path.",
            risk_level=RiskLevel.MEDIUM,
            requires_approval=self.require_approval,
            actions=[
                self._web_recon_action(target, finding, "Refresh recon for observed web risk"),
                self._curl_action(target, finding, "Collect raw HTTP evidence with curl", "get_with_headers", RiskLevel.LOW),
                self._shell_validation_action(
                    target,
                    finding,
                    "Collect raw web risk evidence",
                    'printf "[raw response]\\n"; '
                    'curl -sS -i -L --max-time 10 -A AutoFlow-validation "$TARGET" '
                    '| tee "$ARTIFACT_DIR/web-risk-response.txt" '
                    '| sed -n "1,120p"; '
                    'printf "\\n[risk keywords]\\n"; '
                    'grep -Eio "error|exception|stack|debug|admin|backup|config|secret|token|password|cve|vulnerability|directory listing|index of" '
                    '"$ARTIFACT_DIR/web-risk-response.txt" | sort -u | head -n 80 || true',
                    RiskLevel.MEDIUM,
                    "Collect raw response evidence and highlight likely vulnerability-class keywords.",
                ),
                self._script_probe_action(
                    target,
                    finding,
                    "Summarize generic web risk response",
                    "api_endpoint_probe",
                    RiskLevel.MEDIUM,
                    "Fetch and summarize the response to determine whether a concrete vulnerability class is present.",
                ),
            ],
            success_criteria=["Observation can be mapped to a concrete vulnerability class and reproduction path."],
            failure_criteria=["Observation is generic, informational, or not reproducible."],
            rationale="Generic web risk observations need more context before action selection.",
            metadata={"finding": finding, "category": "web_risk_observation"},
        )

    def _header_check_action(self, target: str, finding: dict) -> TestPlanAction:
        return TestPlanAction(
            name="Confirm response security headers",
            action_kind="script",
            tool="script_runner",
            profile="security_headers_check",
            target=target,
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            expected_impact="One read-only HTTP GET and header parsing.",
            rationale="Confirm the candidate header finding with a controlled read-only check.",
            script_template="security_headers_check",
            metadata={"finding_id": finding.get("id", ""), "validation_role": "confirmatory"},
        )

    def _web_recon_action(self, target: str, finding: dict, name: str) -> TestPlanAction:
        return TestPlanAction(
            name=name,
            action_kind="web_recon",
            tool="web_recon",
            profile="fetch_page",
            target=target,
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            expected_impact="One read-only HTTP GET plus lightweight page parsing.",
            rationale="Collect page or endpoint context before selecting a deeper validation action.",
            metadata={"finding_id": finding.get("id", ""), "validation_role": "context_refresh"},
        )

    def _curl_action(
        self,
        target: str,
        finding: dict,
        name: str,
        profile: str,
        risk_level: RiskLevel,
    ) -> TestPlanAction:
        return TestPlanAction(
            name=name,
            action_kind="tool",
            tool="curl",
            profile=profile,
            target=target,
            risk_level=risk_level,
            requires_approval=False,
            expected_impact="One bounded read-only HTTP request.",
            rationale="Collect raw HTTP evidence with a minimal command-line tool before deeper validation.",
            metadata={"finding_id": finding.get("id", ""), "validation_role": "raw_http_evidence"},
        )

    def _script_probe_action(
        self,
        target: str,
        finding: dict,
        name: str,
        script_template: str,
        risk_level: RiskLevel,
        rationale: str,
    ) -> TestPlanAction:
        return TestPlanAction(
            name=name,
            action_kind="script",
            tool="script_runner",
            profile=script_template,
            target=target,
            risk_level=risk_level,
            requires_approval=self.require_approval,
            expected_impact="Read-only validation probe against the authorized target.",
            rationale=rationale,
            script_template=script_template,
            metadata={
                "finding_id": finding.get("id", ""),
                "validation_role": "active_validation",
                "script_policy_profile": "medium_artifact_script",
            },
        )

    def _shell_validation_action(
        self,
        target: str,
        finding: dict,
        name: str,
        command: str,
        risk_level: RiskLevel,
        rationale: str,
    ) -> TestPlanAction:
        return TestPlanAction(
            name=name,
            action_kind="shell",
            tool="bash_runner",
            profile="bounded_bash",
            target=target,
            risk_level=risk_level,
            requires_approval=self.require_approval if risk_level != RiskLevel.LOW else False,
            expected_impact="Container-only bounded shell validation using read-only requests.",
            rationale=rationale,
            metadata={
                "finding_id": finding.get("id", ""),
                "validation_role": "reproduction_evidence",
                "shell_command": command,
                "shell_policy_profile": "container_lab_shell",
            },
        )

    def _generated_script_action(
        self,
        target: str,
        finding: dict,
        name: str,
        risk_level: RiskLevel,
        script_goal: str,
        rationale: str,
    ) -> TestPlanAction:
        return TestPlanAction(
            name=name,
            action_kind="script",
            tool="script_runner",
            profile="generated_python_script",
            target=target,
            risk_level=risk_level,
            requires_approval=self.require_approval,
            expected_impact="Bounded generated Python validation script scoped to the authorized target.",
            rationale=rationale,
            script_template=None,
            metadata={
                "finding_id": finding.get("id", ""),
                "validation_role": "generated_validation",
                "script_goal": script_goal,
                "script_policy_profile": "medium_artifact_script" if risk_level != RiskLevel.LOW else "low_readonly_http",
                "max_script_attempts": "2",
            },
        )

    def _category(self, finding: dict) -> str:
        metadata = finding.get("metadata") or {}
        return metadata.get("category", "")
