from __future__ import annotations

import json
import re
from typing import Any

from autoflow.observations.models import ToolObservation


NUCLEI_LINE = re.compile(
    r"^\[(?P<template>[^\]]+)\]\s+\[(?P<protocol>[^\]]+)\]\s+\[(?P<severity>[^\]]+)\]\s+(?P<target>\S+)"
)


class ToolObservationParser:
    """把不同工具的 stdout/stderr 统一整理成 ToolObservation。"""

    def parse(
        self,
        *,
        executed_task: dict,
        stdout: str = "",
        stderr: str = "",
    ) -> ToolObservation:
        task = executed_task.get("task", {})
        tool = task.get("tool", "")
        profile = task.get("profile", "")
        target = task.get("target", "")
        raw_result = stdout or executed_task.get("summary", "")
        signals = self._signals_for(tool, raw_result)

        return ToolObservation(
            action_id=executed_task.get("action_id", ""),
            plan_id=task.get("plan_id", ""),
            tool=tool,
            profile=profile,
            target=target,
            status=executed_task.get("status", "completed"),
            risk_level=task.get("risk_level", "low"),
            summary=executed_task.get("summary", ""),
            raw_result=raw_result,
            stderr=stderr or executed_task.get("error", ""),
            artifact_id=executed_task.get("artifact_id"),
            signals=signals,
            metadata={
                "task_type": task.get("type", ""),
                "action_kind": task.get("action_kind", "tool"),
                "source_task": task,
            },
        )

    def _signals_for(self, tool: str, raw_result: str) -> list[dict[str, Any]]:
        if tool == "nuclei":
            return self._parse_nuclei(raw_result)
        if tool == "nikto":
            return self._parse_nikto(raw_result)
        if tool == "script_runner":
            return self._parse_script_runner(raw_result)
        if tool == "web_recon":
            return self._parse_web_recon(raw_result)
        if tool == "whatweb":
            return self._parse_whatweb(raw_result)
        return self._parse_generic(raw_result)

    def _parse_nuclei(self, raw_result: str) -> list[dict[str, Any]]:
        signals = []
        for line in raw_result.splitlines():
            match = NUCLEI_LINE.match(line.strip())
            if not match:
                continue
            signals.append(
                {
                    "kind": "nuclei_template_match",
                    "name": match.group("template"),
                    "severity": match.group("severity"),
                    "protocol": match.group("protocol"),
                    "target": match.group("target"),
                    "evidence": line.strip(),
                }
            )
        return signals

    def _parse_nikto(self, raw_result: str) -> list[dict[str, Any]]:
        signals = []
        for line in raw_result.splitlines():
            text = line.strip()
            if not text.startswith("+"):
                continue
            if text.startswith("+-") or "Target" in text or "Start Time" in text or "End Time" in text:
                continue
            signals.append(
                {
                    "kind": "nikto_observation",
                    "name": self._shorten(text.lstrip("+").strip()),
                    "severity": "info",
                    "evidence": text,
                }
            )
        return signals

    def _parse_script_runner(self, raw_result: str) -> list[dict[str, Any]]:
        payload = self._json_from_text(raw_result)
        if not isinstance(payload, dict):
            return self._parse_generic(raw_result)

        signals = []
        probe = payload.get("probe")
        if probe:
            signals.extend(self._parse_validation_probe(payload))

        missing_headers = payload.get("missing_security_headers") or payload.get("missing_headers") or []
        for header in missing_headers:
            signals.append(
                {
                    "kind": "missing_security_header",
                    "name": str(header),
                    "severity": "low",
                    "evidence": f"Missing response header: {header}",
                }
            )

        headers = payload.get("headers", {})
        cors = headers.get("access-control-allow-origin") or headers.get("Access-Control-Allow-Origin")
        if cors == "*":
            signals.append(
                {
                    "kind": "cors_wildcard",
                    "name": "access-control-allow-origin",
                    "severity": "low",
                    "evidence": "Access-Control-Allow-Origin: *",
                }
            )
        return signals

    def _parse_validation_probe(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        probe = payload.get("probe", "")
        target = payload.get("target", "")
        signals: list[dict[str, Any]] = []
        if probe == "api_endpoint_probe":
            hints = payload.get("sensitivity_hints", [])
            severity = "medium" if hints else "info"
            signals.append(
                {
                    "kind": "api_validation_probe",
                    "name": f"status={payload.get('status')} content_type={payload.get('content_type', '')}",
                    "severity": severity,
                    "target": target,
                    "evidence": json.dumps(
                        {
                            "json_type": payload.get("json_type"),
                            "json_keys": payload.get("json_keys", []),
                            "sensitivity_hints": hints,
                        },
                        ensure_ascii=False,
                    ),
                }
            )
        elif probe == "cors_probe":
            for item in payload.get("results", []):
                allowed_origin = item.get("access_control_allow_origin", "")
                credentials = item.get("access_control_allow_credentials", "")
                severity = "medium" if allowed_origin == "*" or credentials.lower() == "true" else "info"
                signals.append(
                    {
                        "kind": "cors_validation_probe",
                        "name": f"origin={item.get('origin')} allow_origin={allowed_origin}",
                        "severity": severity,
                        "target": target,
                        "evidence": json.dumps(item, ensure_ascii=False),
                    }
                )
        elif probe == "debug_endpoint_probe":
            keywords = payload.get("matched_keywords", [])
            signals.append(
                {
                    "kind": "debug_endpoint_validation_probe",
                    "name": f"matched_keywords={','.join(keywords[:10])}",
                    "severity": "medium" if keywords else "info",
                    "target": target,
                    "evidence": payload.get("sample", ""),
                }
            )
        elif probe == "directory_listing_probe":
            interesting = payload.get("interesting_entries", [])
            signals.append(
                {
                    "kind": "directory_listing_validation_probe",
                    "name": f"entries={payload.get('entry_count', 0)} interesting={len(interesting)}",
                    "severity": "medium" if interesting else "info",
                    "target": target,
                    "evidence": json.dumps({"interesting_entries": interesting}, ensure_ascii=False),
                }
            )
        elif probe == "public_config_probe":
            hints = payload.get("sensitivity_hints", [])
            versions = payload.get("version_hints", [])
            signals.append(
                {
                    "kind": "public_config_validation_probe",
                    "name": f"hints={','.join(hints[:10])}",
                    "severity": "medium" if hints else "info",
                    "target": target,
                    "evidence": json.dumps(
                        {"sensitivity_hints": hints, "version_hints": versions},
                        ensure_ascii=False,
                    ),
                }
            )
        return signals

    def _parse_whatweb(self, raw_result: str) -> list[dict[str, Any]]:
        signals = []
        title = re.search(r"Title\[([^\]]+)\]", raw_result)
        if title:
            signals.append(
                {
                    "kind": "web_title",
                    "name": title.group(1).strip(),
                    "severity": "info",
                    "evidence": title.group(0),
                }
            )
        for header in re.findall(r"UncommonHeaders\[([^\]]+)\]", raw_result):
            signals.append(
                {
                    "kind": "uncommon_headers",
                    "name": header,
                    "severity": "info",
                    "evidence": f"UncommonHeaders[{header}]",
                }
            )
        return signals

    def _parse_web_recon(self, raw_result: str) -> list[dict[str, Any]]:
        payload = self._json_from_text(raw_result)
        if not isinstance(payload, dict):
            return self._parse_generic(raw_result)

        target = payload.get("target", "")
        signals = [
            {
                "kind": "web_recon_page",
                "name": payload.get("title") or target,
                "severity": "info",
                "target": target,
                "evidence": f"status={payload.get('status_code')} title={payload.get('title', '')!r}",
            }
        ]
        for path in payload.get("interesting_paths", [])[:20]:
            signals.append(
                {
                    "kind": "discovered_path",
                    "name": path,
                    "severity": "info",
                    "target": path,
                    "evidence": f"web_recon discovered path {path}",
                }
            )
        robots = payload.get("robots") or {}
        for path in robots.get("interesting_paths", [])[:20]:
            signals.append(
                {
                    "kind": "robots_path",
                    "name": path,
                    "severity": "info",
                    "target": path,
                    "evidence": f"robots.txt referenced {path}",
                }
            )
        for form in payload.get("forms", [])[:20]:
            action = form.get("action") or target
            signals.append(
                {
                    "kind": "web_form",
                    "name": action,
                    "severity": "info",
                    "target": action,
                    "evidence": f"{form.get('method', 'get').upper()} form at {action}",
                }
            )
        return signals

    def _parse_generic(self, raw_result: str) -> list[dict[str, Any]]:
        first_line = next((line.strip() for line in raw_result.splitlines() if line.strip()), "")
        if not first_line:
            return []
        return [{"kind": "raw_output", "name": self._shorten(first_line), "severity": "info", "evidence": first_line}]

    def _json_from_text(self, raw_result: str) -> Any:
        text = raw_result.strip()
        if text.startswith("script_runner completed:"):
            text = text.split("script_runner completed:", 1)[1].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _shorten(self, text: str, limit: int = 90) -> str:
        return text if len(text) <= limit else text[: limit - 3] + "..."
