from __future__ import annotations

import re
import shlex
import socket
from dataclasses import dataclass, field
from urllib.parse import urlparse


URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.I)
FORBIDDEN_SHELL_PATTERNS = [
    ";",
    "&&",
    "||",
    "`",
    "$(",
    "<",
    ">",
    "\n",
    "\r",
]
ALLOWED_VARIABLES = {"$TARGET", "${TARGET}", "$ARTIFACT_DIR", "${ARTIFACT_DIR}"}


@dataclass(frozen=True)
class ShellPolicyProfile:
    name: str
    allowed_commands: set[str] = field(default_factory=set)
    max_length: int = 800
    allow_any_command: bool = False
    allow_shell_control: bool = False
    allow_any_variable: bool = False


SHELL_POLICY_PROFILES = {
    "container_lab_shell": ShellPolicyProfile(
        name="container_lab_shell",
        max_length=4000,
        allow_any_command=True,
        allow_shell_control=True,
        allow_any_variable=True,
    ),
    "low_readonly_http": ShellPolicyProfile(
        name="low_readonly_http",
        allowed_commands={"curl", "grep", "sed", "awk", "jq", "head", "tail", "cut", "sort", "uniq", "wc", "tr"},
        max_length=500,
    ),
    "medium_artifact_shell": ShellPolicyProfile(
        name="medium_artifact_shell",
        allowed_commands={
            "curl",
            "grep",
            "sed",
            "awk",
            "jq",
            "head",
            "tail",
            "cut",
            "sort",
            "uniq",
            "wc",
            "tr",
            "python3",
            "openssl",
        },
        max_length=1000,
    ),
}


@dataclass(frozen=True)
class ShellPolicyDecision:
    allowed: bool
    reason: str = ""


class ShellPolicy:
    """Static checks for bounded bash snippets before container execution."""

    def evaluate(
        self,
        command: str,
        target: str,
        target_scope: list[str],
        profile_name: str = "container_lab_shell",
    ) -> ShellPolicyDecision:
        profile = SHELL_POLICY_PROFILES.get(profile_name)
        if profile is None:
            return ShellPolicyDecision(False, f"Unknown shell policy profile '{profile_name}'")
        command = command.strip()
        if not command:
            return ShellPolicyDecision(False, "Shell command cannot be empty")
        if len(command) > profile.max_length:
            return ShellPolicyDecision(False, f"Shell command exceeds {profile.max_length} characters")
        if not profile.allow_shell_control:
            for pattern in FORBIDDEN_SHELL_PATTERNS:
                if pattern in command:
                    return ShellPolicyDecision(False, f"Shell command contains forbidden pattern '{pattern}'")
        scope_decision = self._target_in_scope(target, target_scope)
        if not scope_decision.allowed:
            return scope_decision
        url_decision = self._urls_in_scope(command, target_scope)
        if not url_decision.allowed:
            return url_decision
        if "$" in command and not profile.allow_any_variable:
            variable_decision = self._variables_allowed(command)
            if not variable_decision.allowed:
                return variable_decision
        if target not in command and "$TARGET" not in command and "${TARGET}" not in command:
            return ShellPolicyDecision(False, "Shell command must reference the provided target or $TARGET")

        if profile.allow_any_command:
            return ShellPolicyDecision(True)

        for segment in command.split("|"):
            try:
                tokens = shlex.split(segment.strip(), posix=True)
            except ValueError as exc:
                return ShellPolicyDecision(False, f"Shell command parse error: {exc}")
            if not tokens:
                return ShellPolicyDecision(False, "Empty shell pipeline segment")
            command_name = tokens[0].rsplit("/", 1)[-1]
            if command_name not in profile.allowed_commands:
                return ShellPolicyDecision(False, f"Shell command '{command_name}' is not allowed")
        return ShellPolicyDecision(True)

    def _variables_allowed(self, command: str) -> ShellPolicyDecision:
        variables = re.findall(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?", command)
        unknown = sorted(set(variables) - ALLOWED_VARIABLES)
        if unknown:
            return ShellPolicyDecision(False, f"Shell command uses unsupported variables: {unknown}")
        return ShellPolicyDecision(True)

    def _urls_in_scope(self, command: str, target_scope: list[str]) -> ShellPolicyDecision:
        allowed_hosts = {self._host(item) for item in target_scope}
        allowed_hosts.discard("")
        for url in URL_RE.findall(command):
            host = self._host(url)
            if host not in allowed_hosts:
                return ShellPolicyDecision(False, f"URL '{url}' is outside authorized scope")
        return ShellPolicyDecision(True)

    def _target_in_scope(self, target: str, target_scope: list[str]) -> ShellPolicyDecision:
        target_host = self._host(target)
        if not target_host:
            return ShellPolicyDecision(False, "Shell target has no host")
        allowed_hosts = {self._host(item) for item in target_scope}
        allowed_hosts.discard("")
        if target_host in allowed_hosts:
            return ShellPolicyDecision(True)

        try:
            target_ip = socket.gethostbyname(target_host)
            allowed_ips = {socket.gethostbyname(host) for host in allowed_hosts}
        except OSError:
            allowed_ips = set()
            target_ip = ""
        if target_ip and target_ip in allowed_ips:
            return ShellPolicyDecision(True)
        return ShellPolicyDecision(False, f"Shell target '{target_host}' is outside authorized scope")

    def _host(self, value: str) -> str:
        parsed = urlparse(value)
        if parsed.hostname:
            return parsed.hostname
        if ":" in value:
            return value.split(":", 1)[0]
        return value
