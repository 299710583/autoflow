from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from autoflow.executor.container_manager import CONTAINER_ARTIFACT_DIR, TaskContainerManager
from autoflow.executor.script_policy import ScriptPolicy
from autoflow.executor.ssh_executor import CommandResult


SCRIPT_IMAGE = "autoflow-kali-tools:latest"


class ScriptRunner:
    """Run approved Python scripts inside a disposable Docker container."""

    def __init__(self, policy: ScriptPolicy | None = None, image: str = SCRIPT_IMAGE) -> None:
        self.policy = policy or ScriptPolicy()
        self.image = image

    def run_template(
        self,
        template: str,
        target: str,
        target_scope: list[str],
        artifact_dir: Path,
        timeout: int = 120,
        policy_profile: str = "low_readonly_http",
    ) -> CommandResult:
        script = self.render_template(template, target)
        return self.run_script(
            script,
            target,
            target_scope,
            artifact_dir,
            timeout=timeout,
            policy_profile=policy_profile,
        )

    def run_script(
        self,
        script: str,
        target: str,
        target_scope: list[str],
        artifact_dir: Path,
        timeout: int = 120,
        policy_profile: str = "low_readonly_http",
    ) -> CommandResult:
        decision = self.policy.evaluate(script, target, target_scope, profile_name=policy_profile)
        if not decision.allowed:
            return CommandResult(
                command=["python3", "script.py"],
                command_text="script_policy_check",
                exit_code=126,
                stdout="",
                stderr=decision.reason,
            )

        artifact_dir.mkdir(parents=True, exist_ok=True)
        script_path = artifact_dir / "script.py"
        script_path.write_text(script, encoding="utf-8")

        with TaskContainerManager(self.image, artifact_dir=artifact_dir) as container:
            return container.exec(["python3", f"{CONTAINER_ARTIFACT_DIR}/script.py"], timeout=timeout)

    def render_template(self, template: str, target: str) -> str:
        if template == "security_headers_check":
            return self._security_headers_check(target)
        if template == "api_endpoint_probe":
            return self._api_endpoint_probe(target)
        if template == "cors_probe":
            return self._cors_probe(target)
        if template == "debug_endpoint_probe":
            return self._debug_endpoint_probe(target)
        if template == "directory_listing_probe":
            return self._directory_listing_probe(target)
        if template == "public_config_probe":
            return self._public_config_probe(target)
        raise ValueError(f"Unknown script template '{template}'")

    def _security_headers_check(self, target: str) -> str:
        return dedent(
            f"""
            import json
            import urllib.request

            target = {target!r}
            request = urllib.request.Request(target, headers={{"User-Agent": "AutoFlow-script-runner/0.1"}})
            result = {{"target": target, "headers": {{}}, "missing": []}}
            expected = [
                "content-security-policy",
                "x-frame-options",
                "x-content-type-options",
                "referrer-policy",
                "permissions-policy",
                "strict-transport-security",
            ]

            with urllib.request.urlopen(request, timeout=10) as response:
                headers = {{key.lower(): value for key, value in response.headers.items()}}
                result["status"] = response.status
                result["headers"] = headers
                result["missing"] = [header for header in expected if header not in headers]

            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            """
        ).strip()

    def _api_endpoint_probe(self, target: str) -> str:
        return dedent(
            f"""
            import json
            import urllib.request

            target = {target!r}
            request = urllib.request.Request(target, headers={{"User-Agent": "AutoFlow-validation/0.1"}})
            result = {{"target": target, "probe": "api_endpoint_probe"}}
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    body = response.read(12000)
                    text = body.decode("utf-8", errors="replace")
                    headers = {{key.lower(): value for key, value in response.headers.items()}}
                    result.update({{
                        "status": response.status,
                        "content_type": headers.get("content-type", ""),
                        "content_length": len(body),
                        "cache_control": headers.get("cache-control", ""),
                        "www_authenticate": headers.get("www-authenticate", ""),
                        "sample": text[:500],
                    }})
                    try:
                        payload = json.loads(text)
                    except Exception:
                        payload = None
                    if isinstance(payload, dict):
                        result["json_type"] = "object"
                        result["json_keys"] = sorted([str(key) for key in payload.keys()])[:50]
                    elif isinstance(payload, list):
                        result["json_type"] = "array"
                        result["json_length"] = len(payload)
                        if payload and isinstance(payload[0], dict):
                            result["json_keys"] = sorted([str(key) for key in payload[0].keys()])[:50]
                    sensitive_words = ["password", "token", "secret", "email", "role", "user", "admin", "key"]
                    lower_text = text.lower()
                    result["sensitivity_hints"] = [word for word in sensitive_words if word in lower_text]
            except Exception as exc:
                result["error"] = str(exc)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            """
        ).strip()

    def _cors_probe(self, target: str) -> str:
        return dedent(
            f"""
            import json
            import urllib.request

            target = {target!r}
            origins = ["https://autoflow.example", "null"]
            results = []
            for origin in origins:
                request = urllib.request.Request(
                    target,
                    headers={{"User-Agent": "AutoFlow-validation/0.1", "Origin": origin}},
                )
                item = {{"origin": origin}}
                try:
                    with urllib.request.urlopen(request, timeout=10) as response:
                        headers = {{key.lower(): value for key, value in response.headers.items()}}
                        item.update({{
                            "status": response.status,
                            "access_control_allow_origin": headers.get("access-control-allow-origin", ""),
                            "access_control_allow_credentials": headers.get("access-control-allow-credentials", ""),
                            "vary": headers.get("vary", ""),
                        }})
                except Exception as exc:
                    item["error"] = str(exc)
                results.append(item)
            print(json.dumps({{"target": target, "probe": "cors_probe", "results": results}}, ensure_ascii=False, sort_keys=True))
            """
        ).strip()

    def _debug_endpoint_probe(self, target: str) -> str:
        return dedent(
            f"""
            import json
            import urllib.request

            target = {target!r}
            request = urllib.request.Request(target, headers={{"User-Agent": "AutoFlow-validation/0.1"}})
            result = {{"target": target, "probe": "debug_endpoint_probe"}}
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    body = response.read(16000)
                    text = body.decode("utf-8", errors="replace")
                    headers = {{key.lower(): value for key, value in response.headers.items()}}
                    keywords = [
                        "process", "heap", "memory", "uptime", "nodejs", "express", "stack",
                        "trace", "exception", "env", "secret", "token", "password", "prometheus",
                    ]
                    lower_text = text.lower()
                    result.update({{
                        "status": response.status,
                        "content_type": headers.get("content-type", ""),
                        "content_length": len(body),
                        "matched_keywords": [word for word in keywords if word in lower_text],
                        "sample": text[:700],
                    }})
            except Exception as exc:
                result["error"] = str(exc)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            """
        ).strip()

    def _directory_listing_probe(self, target: str) -> str:
        return dedent(
            f"""
            import json
            import re
            import urllib.request

            target = {target!r}
            request = urllib.request.Request(target, headers={{"User-Agent": "AutoFlow-validation/0.1"}})
            result = {{"target": target, "probe": "directory_listing_probe"}}
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    body = response.read(20000)
                    text = body.decode("utf-8", errors="replace")
                    names = re.findall(r'href=["\\']([^"\\']+)["\\']', text, flags=re.I)
                    interesting = []
                    patterns = [
                        ".bak", ".backup", ".old", ".zip", ".tar", ".gz", ".db", ".sqlite",
                        ".kdbx", ".key", ".pem", ".log", ".env", "config", "secret", "password",
                    ]
                    for name in names:
                        lower_name = name.lower()
                        if any(pattern in lower_name for pattern in patterns):
                            interesting.append(name)
                    result.update({{
                        "status": response.status,
                        "entry_count": len(names),
                        "entries": names[:100],
                        "interesting_entries": interesting[:100],
                        "sample": text[:700],
                    }})
            except Exception as exc:
                result["error"] = str(exc)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            """
        ).strip()

    def _public_config_probe(self, target: str) -> str:
        return dedent(
            f"""
            import json
            import re
            import urllib.request

            target = {target!r}
            request = urllib.request.Request(target, headers={{"User-Agent": "AutoFlow-validation/0.1"}})
            result = {{"target": target, "probe": "public_config_probe"}}
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    body = response.read(20000)
                    text = body.decode("utf-8", errors="replace")
                    lower_text = text.lower()
                    hints = []
                    for word in ["password", "secret", "token", "apikey", "api_key", "private", "database", "mongodb", "redis"]:
                        if word in lower_text:
                            hints.append(word)
                    versions = re.findall(r'"(?:version|node|express|angular)"\\s*:\\s*"([^"]+)"', text, flags=re.I)
                    result.update({{
                        "status": response.status,
                        "content_type": response.headers.get("content-type", ""),
                        "content_length": len(body),
                        "sensitivity_hints": hints,
                        "version_hints": versions[:50],
                        "sample": text[:700],
                    }})
            except Exception as exc:
                result["error"] = str(exc)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            """
        ).strip()
