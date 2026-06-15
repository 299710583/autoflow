from __future__ import annotations

import ast
import socket
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from urllib.parse import urlparse


ALLOWED_IMPORT_ROOTS = {
    "json",
    "re",
    "ssl",
    "socket",
    "sys",
    "time",
    "urllib",
}


@dataclass(frozen=True)
class ScriptPolicyProfile:
    name: str
    allowed_import_roots: set[str]
    allow_open_artifact_writes: bool = False
    allow_subprocess: bool = False
    allowed_subprocess_commands: set[str] = field(default_factory=set)


SCRIPT_POLICY_PROFILES = {
    "low_readonly_http": ScriptPolicyProfile(
        name="low_readonly_http",
        allowed_import_roots=ALLOWED_IMPORT_ROOTS,
    ),
    "medium_artifact_script": ScriptPolicyProfile(
        name="medium_artifact_script",
        allowed_import_roots={
            *ALLOWED_IMPORT_ROOTS,
            "base64",
            "hashlib",
            "pathlib",
            "subprocess",
        },
        allow_open_artifact_writes=True,
        allow_subprocess=True,
        allowed_subprocess_commands={"curl", "openssl", "python3"},
    ),
    "high_lab_poc": ScriptPolicyProfile(
        name="high_lab_poc",
        allowed_import_roots={
            *ALLOWED_IMPORT_ROOTS,
            "base64",
            "hashlib",
            "pathlib",
            "subprocess",
        },
        allow_open_artifact_writes=True,
        allow_subprocess=True,
        allowed_subprocess_commands={
            "curl",
            "ffuf",
            "gobuster",
            "nmap",
            "nuclei",
            "openssl",
            "python3",
            "sqlmap",
        },
    ),
}

FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
}

FORBIDDEN_ATTRS = {
    "system",
    "popen",
    "spawn",
    "fork",
    "execv",
    "execve",
    "remove",
    "unlink",
    "rmdir",
    "rename",
}


@dataclass(frozen=True)
class ScriptPolicyDecision:
    allowed: bool
    reason: str = ""


class ScriptPolicy:
    """Static checks for constrained Python scripts before they enter Docker."""

    def evaluate(
        self,
        script: str,
        target: str,
        target_scope: list[str],
        profile_name: str = "low_readonly_http",
    ) -> ScriptPolicyDecision:
        profile = SCRIPT_POLICY_PROFILES.get(profile_name)
        if profile is None:
            return ScriptPolicyDecision(False, f"Unknown script policy profile '{profile_name}'")

        try:
            tree = ast.parse(script)
        except SyntaxError as exc:
            return ScriptPolicyDecision(False, f"Script syntax error: {exc}")

        scope_decision = self._target_in_scope(target, target_scope)
        if not scope_decision.allowed:
            return scope_decision

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                decision = self._check_import(node, profile)
                if not decision.allowed:
                    return decision
            elif isinstance(node, ast.Call):
                decision = self._check_call(node, profile)
                if not decision.allowed:
                    return decision

        return ScriptPolicyDecision(True)

    def _check_import(
        self,
        node: ast.Import | ast.ImportFrom,
        profile: ScriptPolicyProfile,
    ) -> ScriptPolicyDecision:
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif node.module:
            names = [node.module]

        for name in names:
            root = name.split(".", 1)[0]
            if root not in profile.allowed_import_roots:
                return ScriptPolicyDecision(
                    False,
                    f"Import '{name}' is not allowed by profile '{profile.name}'",
                )
        return ScriptPolicyDecision(True)

    def _check_call(self, node: ast.Call, profile: ScriptPolicyProfile) -> ScriptPolicyDecision:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            return ScriptPolicyDecision(False, f"Call '{node.func.id}' is not allowed")
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            return self._check_open_call(node, profile)
        if self._is_subprocess_call(node):
            return self._check_subprocess_call(node, profile)
        if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_ATTRS:
            return ScriptPolicyDecision(False, f"Call '*.{node.func.attr}' is not allowed")
        return ScriptPolicyDecision(True)

    def _check_open_call(self, node: ast.Call, profile: ScriptPolicyProfile) -> ScriptPolicyDecision:
        if not profile.allow_open_artifact_writes:
            return ScriptPolicyDecision(False, f"Call 'open' is not allowed by profile '{profile.name}'")
        if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
            return ScriptPolicyDecision(False, "open path must be a static string")

        raw_path = node.args[0].value
        path = PurePosixPath(raw_path)
        if not path.is_absolute() or not str(path).startswith("/work/artifacts/"):
            return ScriptPolicyDecision(False, "open is restricted to /work/artifacts")

        mode = "r"
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
            mode = node.args[1].value
        for keyword in node.keywords:
            if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                mode = str(keyword.value.value)
        if "r" in mode and all(flag not in mode for flag in ("w", "a", "x", "+")):
            return ScriptPolicyDecision(False, "open may only write artifact files")
        return ScriptPolicyDecision(True)

    def _is_subprocess_call(self, node: ast.Call) -> bool:
        return (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        )

    def _check_subprocess_call(
        self,
        node: ast.Call,
        profile: ScriptPolicyProfile,
    ) -> ScriptPolicyDecision:
        if not profile.allow_subprocess:
            return ScriptPolicyDecision(False, f"subprocess is not allowed by profile '{profile.name}'")
        if not isinstance(node.func, ast.Attribute) or node.func.attr not in {"run", "check_output"}:
            return ScriptPolicyDecision(False, "Only subprocess.run/check_output are allowed")
        for keyword in node.keywords:
            if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                return ScriptPolicyDecision(False, "subprocess shell=True is not allowed")
        if not node.args:
            return ScriptPolicyDecision(False, "subprocess command must be static")
        command = node.args[0]
        command_name = self._static_command_name(command)
        if not command_name:
            return ScriptPolicyDecision(False, "subprocess command must be a static list")
        if command_name not in profile.allowed_subprocess_commands:
            return ScriptPolicyDecision(False, f"subprocess command '{command_name}' is not allowed")
        return ScriptPolicyDecision(True)

    def _static_command_name(self, node: ast.AST) -> str:
        if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
            return ""
        first = node.elts[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return PurePosixPath(first.value).name
        return ""

    def _target_in_scope(self, target: str, target_scope: list[str]) -> ScriptPolicyDecision:
        target_host = self._host(target)
        if not target_host:
            return ScriptPolicyDecision(False, "Script target has no host")
        allowed_hosts = {self._host(item) for item in target_scope}
        allowed_hosts.discard("")
        if target_host in allowed_hosts:
            return ScriptPolicyDecision(True)

        try:
            target_ip = socket.gethostbyname(target_host)
            allowed_ips = {socket.gethostbyname(host) for host in allowed_hosts}
        except OSError:
            allowed_ips = set()
            target_ip = ""
        if target_ip and target_ip in allowed_ips:
            return ScriptPolicyDecision(True)

        return ScriptPolicyDecision(False, f"Script target '{target_host}' is outside authorized scope")

    def _host(self, value: str) -> str:
        parsed = urlparse(value)
        if parsed.hostname:
            return parsed.hostname
        if ":" in value:
            return value.split(":", 1)[0]
        return value
