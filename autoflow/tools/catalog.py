from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autoflow.executor.script_runner import ScriptRunner
from autoflow.executor.tool_registry import ToolRegistry
from autoflow.tools.manifest import ToolManifestRegistry


SCRIPT_TOOL_TEMPLATES = {
    "security_headers_check": "Check browser-facing security headers and CORS wildcard behavior.",
    "api_endpoint_probe": "Fetch an API endpoint and summarize status, JSON shape, keys, and sensitivity hints.",
    "cors_probe": "Send read-only requests with controlled Origin headers and summarize CORS behavior.",
    "debug_endpoint_probe": "Fetch a debug or metrics endpoint and classify runtime information exposure.",
    "directory_listing_probe": "Fetch a directory listing page and classify interesting filenames.",
    "public_config_probe": "Fetch a public config-like file and classify secrets, endpoints, and version hints.",
}

VALIDATION_HELPER_PROFILES = {
    ("curl", "headers"),
    ("curl", "get"),
    ("curl", "get_with_headers"),
    ("nuclei", "discovery_all_severity"),
    ("nuclei", "discovery_low"),
    ("nuclei", "discovery_medium"),
    ("nikto", "basic_web_check"),
    ("whatweb", "web_fingerprint"),
    ("httpx", "basic_probe"),
    ("wafw00f", "detect_waf"),
    ("script_runner", "security_headers_check"),
    ("script_runner", "api_endpoint_probe"),
    ("script_runner", "cors_probe"),
    ("script_runner", "debug_endpoint_probe"),
    ("script_runner", "directory_listing_probe"),
    ("script_runner", "public_config_probe"),
}


@dataclass(frozen=True)
class ToolFunction:
    name: str
    description: str
    parameters: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolCatalog:
    """Build function-calling schemas from AutoFlow tools and built-in helpers."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        script_runner: ScriptRunner | None = None,
        manifest: ToolManifestRegistry | None = None,
    ) -> None:
        self.registry = registry or ToolRegistry.from_file()
        self.script_runner = script_runner or ScriptRunner()
        self.manifest = manifest or ToolManifestRegistry()

    def functions(self, phases: set[str] | None = None) -> list[ToolFunction]:
        functions: list[ToolFunction] = [
            self._read_agent_memory_tool(),
            self._list_known_targets_tool(),
            self._search_observations_tool(),
        ]
        if self._phase_allows_builtin(phases, {"discovery", "validation"}):
            functions.append(self._web_recon_tool())
        if self._phase_allows_builtin(phases, {"validation", "artifact_audit"}):
            functions.append(self._bounded_shell_tool())
        if self._phase_allows_builtin(phases, {"validation"}):
            functions.append(self._custom_validation_script_tool())
        functions.extend(self._container_tool_functions(phases))
        functions.extend(self._script_tool_functions(phases))
        return functions

    def openai_tools(self, phases: set[str] | None = None) -> list[dict[str, Any]]:
        return [function.openai_schema() for function in self.functions(phases)]

    def function_names(self) -> set[str]:
        return {function.name for function in self.functions()}

    def _container_tool_functions(self, phases: set[str] | None = None) -> list[ToolFunction]:
        functions: list[ToolFunction] = []
        for tool_name, tool in sorted(self.registry.tools.items()):
            if not tool.enabled:
                continue
            for profile_name, profile in sorted(tool.profiles.items()):
                if not self._profile_allowed_for_phase(tool_name, profile_name, phases):
                    continue
                properties: dict[str, Any] = {}
                required: list[str] = []
                for arg in profile.allowed_args:
                    if arg == "output":
                        continue
                    properties[arg] = {
                        "type": "string",
                        "description": self._arg_description(arg),
                    }
                    required.append(arg)
                if profile.target_required and "target" not in properties:
                    properties["target"] = {"type": "string", "description": self._arg_description("target")}
                    required.append("target")
                functions.append(
                    ToolFunction(
                        name=self.container_function_name(tool_name, profile_name),
                        description=self._container_tool_description(
                            tool_name=tool_name,
                            profile_name=profile_name,
                            risk_level=profile.risk or tool.risk,
                            profile_description=profile.description,
                            target_required=profile.target_required,
                        ),
                        parameters={
                            "type": "object",
                            "properties": properties,
                            "required": sorted(set(required)),
                            "additionalProperties": False,
                        },
                        metadata={
                            "kind": "container_tool",
                            "tool": tool_name,
                            "profile": profile_name,
                            "risk_level": profile.risk or tool.risk,
                        },
                    )
                )
        return functions

    def _script_tool_functions(self, phases: set[str] | None = None) -> list[ToolFunction]:
        return [
            ToolFunction(
                name=self.script_function_name(template),
                description=self._script_tool_description(template, description),
                parameters={
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Authorized HTTP/HTTPS target or endpoint.",
                        }
                    },
                    "required": ["target"],
                    "additionalProperties": False,
                },
                metadata={
                    "kind": "script_template",
                    "tool": "script_runner",
                    "profile": template,
                    "risk_level": "medium" if template != "security_headers_check" else "low",
                },
            )
            for template, description in sorted(SCRIPT_TOOL_TEMPLATES.items())
            if self._profile_allowed_for_phase("script_runner", template, phases)
        ]

    def _profile_allowed_for_phase(self, tool_name: str, profile_name: str, phases: set[str] | None) -> bool:
        if phases is None:
            return True
        if "validation" in phases and (tool_name, profile_name) in VALIDATION_HELPER_PROFILES:
            return True
        entries = self.manifest.by_profile(tool_name, profile_name)
        if not entries:
            return False
        return any(str(entry.get("phase", "")) in phases for entry in entries)

    def _phase_allows_builtin(self, phases: set[str] | None, allowed: set[str]) -> bool:
        return phases is None or bool(phases & allowed)

    def _web_recon_tool(self) -> ToolFunction:
        return ToolFunction(
            name="web_recon_fetch_page",
            description="Fetch and parse an authorized web page. Returns title, links, forms, scripts, robots and sitemap context.",
            parameters={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Authorized HTTP/HTTPS target URL."}
                },
                "required": ["target"],
                "additionalProperties": False,
            },
            metadata={"kind": "web_recon", "tool": "web_recon", "profile": "fetch_page", "risk_level": "low"},
        )

    def _bounded_shell_tool(self) -> ToolFunction:
        return ToolFunction(
            name="run_shell__bounded_bash",
            description=(
                "Run a bounded bash pipeline inside the disposable AutoFlow tool container, never on the host. "
                "Use for compact curl/grep/jq/head style validation. The command must reference $TARGET or the target URL."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Authorized target URL."},
                    "command": {
                        "type": "string",
                        "description": "Single bash pipeline using allowed commands; prefer $TARGET for the target.",
                    },
                    "policy_profile": {
                        "type": "string",
                        "description": "Shell policy profile: container_lab_shell, low_readonly_http, or medium_artifact_shell.",
                    },
                },
                "required": ["target", "command"],
                "additionalProperties": False,
            },
            metadata={"kind": "shell", "tool": "bash_runner", "profile": "bounded_bash", "risk_level": "medium"},
        )

    def _custom_validation_script_tool(self) -> ToolFunction:
        return ToolFunction(
            name="run_script__custom_validation",
            description=(
                "Run a custom Python validation script written by the LLM inside the disposable AutoFlow Docker "
                "tool container, never on the host. Use this when existing tool profiles do not answer a specific "
                "validation question. The script is statically checked by ScriptPolicy, receives TARGET, "
                "TARGET_SCOPE, and ARTIFACT_DIR variables, should use short timeouts, and should print one concise "
                "JSON object with evidence."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Authorized HTTP/HTTPS target or endpoint."},
                    "script_source": {
                        "type": "string",
                        "description": (
                            "Complete Python 3 script body. Prefer urllib/json/re/time/socket/ssl. "
                            "Use the provided TARGET variable instead of inventing targets. Print one JSON object."
                        ),
                    },
                    "policy_profile": {
                        "type": "string",
                        "enum": ["low_readonly_http", "medium_artifact_script", "high_lab_poc"],
                        "description": (
                            "Script policy profile. Prefer low_readonly_http for HTTP evidence; use "
                            "medium_artifact_script only when artifact writes or approved subprocess helpers are needed."
                        ),
                    },
                    "timeout": {
                        "type": "string",
                        "description": "Maximum runtime seconds, normally 30-120.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this custom script is needed instead of an existing tool.",
                    },
                },
                "required": ["target", "script_source"],
                "additionalProperties": False,
            },
            metadata={
                "kind": "custom_script",
                "tool": "script_runner",
                "profile": "custom_validation",
                "risk_level": "medium",
            },
        )

    def _read_agent_memory_tool(self) -> ToolFunction:
        return ToolFunction(
            name="read_agent_memory",
            description="Read the compact AutoFlow memory pack for this assessment.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            metadata={"kind": "memory"},
        )

    def _list_known_targets_tool(self) -> ToolFunction:
        return ToolFunction(
            name="list_known_targets",
            description="List authorized and discovered targets currently known to AutoFlow.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            metadata={"kind": "memory"},
        )

    def _search_observations_tool(self) -> ToolFunction:
        return ToolFunction(
            name="search_observations",
            description="Search existing tool observations by keyword.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Keyword or signal kind to search."}},
                "required": ["query"],
                "additionalProperties": False,
            },
            metadata={"kind": "memory"},
        )

    @staticmethod
    def container_function_name(tool: str, profile: str) -> str:
        return f"run_{tool}__{profile}"

    @staticmethod
    def script_function_name(template: str) -> str:
        return f"run_script__{template}"

    def _container_tool_description(
        self,
        tool_name: str,
        profile_name: str,
        risk_level: str,
        profile_description: str | None = None,
        target_required: bool = True,
    ) -> str:
        specific = {
            ("subfinder", "passive_domain_enum"): (
                "Passive subdomain enumeration for an authorized root domain. "
                "Use only when the target is a domain, not an IP address or URL."
            ),
            ("feroxbuster", "small_directory_check"): (
                "Small, rate-limited web path discovery for authorized HTTP/HTTPS targets."
            ),
            ("sqlmap", "basic_get_param_check"): (
                "Bounded SQL injection validation for an authorized URL with query parameters. "
                "Use during validation when prior observations suggest injectable parameters."
            ),
            ("hydra", "single_credential_check"): (
                "Single known credential validation only; this profile is not for wordlist brute force."
            ),
            ("medusa", "single_credential_check"): (
                "Single known credential validation only; this profile is not for wordlist brute force."
            ),
            ("trivy", "filesystem_audit"): (
                "Scans an AutoFlow artifact/source path mounted into the container."
            ),
            ("bandit", "python_source_audit"): (
                "Scans an AutoFlow artifact/source path mounted into the container."
            ),
            ("gitleaks", "secret_scan"): (
                "Scans an AutoFlow artifact/source path mounted into the container."
            ),
            ("semgrep", "source_audit"): (
                "Scans an AutoFlow artifact/source path mounted into the container."
            ),
        }.get((tool_name, profile_name))
        scope = "authorized target" if target_required else "AutoFlow artifact/source path"
        base = (
            f"Run {tool_name}/{profile_name} inside the disposable AutoFlow Docker tool container "
            f"(autoflow-kali-tools), never on the host shell. Risk={risk_level}. "
            f"Scope={scope}. Output is saved as an artifact and summarized for the agent memory."
        )
        details = " ".join(part for part in (profile_description, specific) if part)
        manifest_details = self._manifest_details(tool_name, profile_name)
        if manifest_details:
            details = " ".join(part for part in (details, manifest_details) if part)
        if details:
            return f"{base} {details}"
        return base

    def _script_tool_description(self, template: str, fallback: str) -> str:
        base = f"Run script_runner template '{template}' inside the disposable AutoFlow tool container."
        details = self._manifest_details("script_runner", template)
        return f"{base} {details or fallback}"

    def _manifest_details(self, tool_name: str, profile_name: str) -> str:
        entries = self.manifest.by_profile(tool_name, profile_name)
        if not entries:
            return ""
        phases = ", ".join(sorted({str(entry.get("phase", "")) for entry in entries if entry.get("phase")}))
        purposes = "; ".join(str(entry.get("purpose", "")) for entry in entries if entry.get("purpose"))
        best_for = []
        avoid_when = []
        for entry in entries:
            best_for.extend(str(value) for value in entry.get("best_for", [])[:3])
            avoid_when.extend(str(value) for value in entry.get("avoid_when", [])[:3])
        parts = []
        if phases:
            parts.append(f"Manifest phases: {phases}.")
        if purposes:
            parts.append(f"Purpose: {purposes}")
        if best_for:
            parts.append(f"Best for: {', '.join(best_for[:5])}.")
        if avoid_when:
            parts.append(f"Avoid when: {', '.join(avoid_when[:5])}.")
        return " ".join(parts)

    @staticmethod
    def _arg_description(arg: str) -> str:
        descriptions = {
            "target": (
                "Authorized target host, URL, endpoint, or domain as required by this tool profile. "
                "Execution is scoped by AutoFlow before entering the container."
            ),
            "port": "Authorized target port.",
            "maxtime": "Maximum runtime in seconds.",
            "service": "Network service module name expected by the tool, for example ssh, ftp, smb, or http-form-post.",
            "username": "Single known username to validate.",
            "password": "Single known password to validate.",
            "path": (
                "Project-local AutoFlow artifact/source path to scan inside the container. "
                "Allowed roots are data/artifacts, data/source, and data/source_audit."
            ),
        }
        return descriptions.get(arg, f"Tool argument '{arg}'.")
