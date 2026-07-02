from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from autoflow.graph.state import AutoFlowState
from autoflow.settings import settings


@dataclass(frozen=True)
class CompressionLimits:
    observations: int
    web_context: int
    validation_history: int
    recent_tool_results: int
    signals: int
    text: int
    artifact_refs: int


COMPRESSION_LIMITS: dict[str, CompressionLimits] = {
    "level_0": CompressionLimits(20, 8, 10, 8, 10, 1000, 20),
    "level_1": CompressionLimits(12, 5, 8, 5, 6, 700, 16),
    "level_2": CompressionLimits(8, 3, 5, 3, 4, 500, 12),
    "level_3": CompressionLimits(4, 2, 3, 1, 3, 350, 8),
    "level_4": CompressionLimits(2, 1, 2, 1, 2, 250, 5),
}


class MemoryCompressor:
    """Build task-focused memory views for high-context agents.

    The first supported view is validation_react. It deterministically filters
    AutoFlowState and persisted memory around the current finding, then trims the
    result according to a simple token budget estimate.
    """

    def __init__(
        self,
        *,
        context_window_tokens: int | None = None,
        reserved_output_tokens: int | None = None,
        base_overhead_tokens: int = 12000,
    ) -> None:
        self.context_window_tokens = context_window_tokens or settings.memory_context_window_tokens
        self.reserved_output_tokens = reserved_output_tokens or settings.memory_reserved_output_tokens
        self.base_overhead_tokens = base_overhead_tokens

    def build_for_agent(
        self,
        state: AutoFlowState,
        *,
        agent_name: str,
        focus: dict[str, Any] | None = None,
        base_memory: dict[str, Any] | None = None,
        validation_plan: dict[str, Any] | None = None,
        recent_tool_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if agent_name != "validation_react":
            return self._fallback_view(state, agent_name, focus or {}, base_memory or {})
        return self._build_validation_react_view(
            state,
            focus=focus or {},
            base_memory=base_memory or {},
            validation_plan=validation_plan or {},
            recent_tool_results=recent_tool_results or [],
        )

    def _build_validation_react_view(
        self,
        state: AutoFlowState,
        *,
        focus: dict[str, Any],
        base_memory: dict[str, Any],
        validation_plan: dict[str, Any],
        recent_tool_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        focus = self._normalized_focus(focus, state)
        candidates = self._validation_candidates(state, base_memory, focus, recent_tool_results)
        level = self._select_level(candidates, focus)
        view = self._materialize_validation_view(
            candidates,
            focus=focus,
            validation_plan=validation_plan,
            base_memory=base_memory,
            limits=COMPRESSION_LIMITS[level],
            level=level,
        )
        view["budget_report"] = self._budget_report(view, agent_name="validation_react", level=level)
        return view

    def _validation_candidates(
        self,
        state: AutoFlowState,
        base_memory: dict[str, Any],
        focus: dict[str, Any],
        recent_tool_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        observations = self._dedupe_dicts(
            [
                *self._dict_list(base_memory.get("tool_observations")),
                *self._dict_list(state.get("tool_observations")),
            ],
            key_fields=("id", "action_id", "tool", "profile", "target", "summary"),
        )
        scored_observations = self._rank_related(observations, focus)

        web_context = self._dedupe_dicts(
            [
                *self._dict_list(base_memory.get("web_context")),
                *self._dict_list(state.get("web_recon")),
            ],
            key_fields=("target",),
        )
        scored_web_context = self._rank_related(web_context, focus)

        validation_history = self._dedupe_dicts(
            [
                *self._dict_list(state.get("validation_results")),
                *self._dict_list(state.get("validation_react_results")),
            ],
            key_fields=("id", "finding_id", "validation_plan_id", "target", "reasoning"),
        )
        scored_validation_history = self._rank_related(validation_history, focus)

        executed_tasks = self._dedupe_dicts(
            [
                *self._dict_list(base_memory.get("executed_actions")),
                *self._dict_list(base_memory.get("failed_actions")),
                *self._dict_list(state.get("executed_tasks")),
            ],
            key_fields=("action_id", "tool", "profile", "target", "summary", "error"),
        )
        scored_executed_tasks = self._rank_related(executed_tasks, focus)

        scored_recent_results = self._rank_related(self._dict_list(recent_tool_results), focus)

        return {
            "current_finding": focus.get("finding", {}),
            "observations": scored_observations,
            "web_context": scored_web_context,
            "validation_history": scored_validation_history,
            "executed_tasks": scored_executed_tasks,
            "recent_tool_results": scored_recent_results,
            "all_observation_count": len(observations),
            "all_web_context_count": len(web_context),
            "all_validation_history_count": len(validation_history),
            "all_executed_task_count": len(executed_tasks),
        }

    def _materialize_validation_view(
        self,
        candidates: dict[str, Any],
        *,
        focus: dict[str, Any],
        validation_plan: dict[str, Any],
        base_memory: dict[str, Any],
        limits: CompressionLimits,
        level: str,
    ) -> dict[str, Any]:
        observations = [
            self._compact_observation(item, limits)
            for item in self._take_ranked(candidates["observations"], limits.observations)
        ]
        web_context = [
            self._compact_web_context(item, limits)
            for item in self._take_ranked(candidates["web_context"], limits.web_context)
        ]
        validation_history = [
            self._compact_validation_history(item, limits)
            for item in self._take_ranked(candidates["validation_history"], limits.validation_history)
        ]
        recent_tool_results = [
            self._compact_tool_result(item, limits)
            for item in self._take_ranked(candidates["recent_tool_results"], limits.recent_tool_results)
        ]
        executed_tasks = [
            self._compact_executed_task(item, limits)
            for item in self._take_ranked(candidates["executed_tasks"], limits.validation_history)
        ]
        evidence_ledger = self._evidence_ledger(
            observations=observations,
            validation_history=validation_history,
            executed_tasks=executed_tasks,
            recent_tool_results=recent_tool_results,
            focus=focus,
            limits=limits,
        )
        artifact_refs = self._artifact_refs(
            observations=observations,
            validation_history=validation_history,
            executed_tasks=executed_tasks,
            recent_tool_results=recent_tool_results,
            limit=limits.artifact_refs,
        )

        return {
            "view_type": "validation_react",
            "compression_level": level,
            "focus": {
                "finding_id": focus.get("finding_id"),
                "target": focus.get("target"),
                "host": focus.get("host"),
                "path": focus.get("path"),
                "category": focus.get("category"),
                "title": focus.get("title"),
            },
            "base_memory_summary": {
                "project_id": base_memory.get("project_id"),
                "flow_id": base_memory.get("flow_id"),
                "target_scope": base_memory.get("target_scope", [])[:10],
                "user_prompt": self._trim(base_memory.get("user_prompt", ""), limits.text),
                "rules_of_engagement": base_memory.get("rules_of_engagement", {}),
                "memory_counts": base_memory.get("memory_counts", {}),
            },
            "current_finding": self._compact_finding(focus.get("finding", {}), limits),
            "validation_plan": self._compact_validation_plan(validation_plan, limits),
            "related_observations": observations,
            "related_web_context": web_context,
            "related_validation_history": validation_history,
            "recent_tool_results": recent_tool_results,
            "evidence_ledger": evidence_ledger,
            "artifact_refs": artifact_refs,
            "excluded_counts": {
                "observations": max(0, candidates["all_observation_count"] - len(observations)),
                "web_context": max(0, candidates["all_web_context_count"] - len(web_context)),
                "validation_history": max(0, candidates["all_validation_history_count"] - len(validation_history)),
                "executed_tasks": max(0, candidates["all_executed_task_count"] - len(executed_tasks)),
            },
        }

    def _select_level(self, candidates: dict[str, Any], focus: dict[str, Any]) -> str:
        probe = {
            "focus": focus,
            "observations": [item for _, item in candidates["observations"][:20]],
            "web_context": [item for _, item in candidates["web_context"][:8]],
            "validation_history": [item for _, item in candidates["validation_history"][:10]],
            "recent_tool_results": [item for _, item in candidates["recent_tool_results"][:8]],
            "executed_tasks": [item for _, item in candidates["executed_tasks"][:10]],
        }
        usage_ratio = self._usage_ratio(probe)
        if usage_ratio >= 0.90:
            return "level_4"
        if usage_ratio >= 0.80:
            return "level_3"
        if usage_ratio >= 0.65:
            return "level_2"
        if usage_ratio >= 0.50:
            return "level_1"
        return "level_0"

    def _normalized_focus(self, focus: dict[str, Any], state: AutoFlowState) -> dict[str, Any]:
        finding = self._dict_or_empty(focus.get("finding"))
        if not finding and focus.get("finding_id"):
            finding = next(
                (
                    item
                    for item in self._dict_list(state.get("findings"))
                    if item.get("id") == focus.get("finding_id")
                ),
                {},
            )
        metadata = self._dict_or_empty(finding.get("metadata"))
        target = str(focus.get("target") or finding.get("target") or "")
        parsed = urlparse(target if "://" in target else f"//{target}")
        return {
            "finding": finding,
            "finding_id": str(focus.get("finding_id") or finding.get("id") or ""),
            "target": target,
            "host": parsed.hostname or "",
            "path": parsed.path or "",
            "category": str(focus.get("category") or metadata.get("category") or ""),
            "title": str(focus.get("title") or finding.get("title") or ""),
        }

    def _rank_related(self, items: list[dict[str, Any]], focus: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
        ranked = []
        for index, item in enumerate(items):
            score = self._score_related(item, focus)
            if score <= 0:
                continue
            score += min(index, 20)
            ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return ranked

    def _score_related(self, item: dict[str, Any], focus: dict[str, Any]) -> int:
        text = self._item_text(item)
        score = 0
        finding_id = str(focus.get("finding_id") or "")
        target = str(focus.get("target") or "").lower()
        host = str(focus.get("host") or "").lower()
        path = str(focus.get("path") or "").lower()
        category = str(focus.get("category") or "").lower()
        if finding_id and finding_id.lower() in text:
            score += 100
        item_target = str(item.get("target") or "").lower()
        if target and (target == item_target or target in text):
            score += 80
        if path and path != "/" and path in text:
            score += 70
        if host and host in text:
            score += 40
        if category and category.replace("_", " ") in text.replace("_", " "):
            score += 35
        if any(keyword in text for keyword in self._category_keywords(category)):
            score += 25
        if str(item.get("status", "")).lower() in {"failed", "error"}:
            score += 15
        return score

    def _category_keywords(self, category: str) -> list[str]:
        mapping = {
            "api_exposure": ["api", "/api/", "/rest/", "json", "unauthenticated"],
            "sql_injection": ["sql", "sqli", "injection", "sqlmap"],
            "cors": ["cors", "access-control-allow-origin", "origin"],
            "directory_listing": ["directory", "listing", "index of", "href"],
            "public_config": ["config", "package.json", ".env", "secret", "dependencies"],
            "debug_endpoint": ["debug", "metrics", "stack", "trace"],
        }
        return mapping.get(category, [part for part in category.replace("_", " ").split() if len(part) > 2])

    def _evidence_ledger(
        self,
        *,
        observations: list[dict[str, Any]],
        validation_history: list[dict[str, Any]],
        executed_tasks: list[dict[str, Any]],
        recent_tool_results: list[dict[str, Any]],
        focus: dict[str, Any],
        limits: CompressionLimits,
    ) -> dict[str, list[str]]:
        texts = [
            *(self._observation_lines(item) for item in observations),
            *(self._history_lines(item) for item in validation_history),
            *(self._task_lines(item) for item in executed_tasks),
            *(self._tool_result_lines(item) for item in recent_tool_results),
        ]
        lines = [line for group in texts for line in group]
        confirmed = []
        negative = []
        failed = []
        missing = []
        do_not_repeat = []
        for line in lines:
            lowered = line.lower()
            if any(word in lowered for word in ["missing_evidence", "missing evidence", "need ", "缺少"]):
                missing.append(line)
            elif any(word in lowered for word in ["failed", "error", "timeout", "exception"]):
                failed.append(line)
            elif any(word in lowered for word in [" no ", "not ", "absent", "false_positive", "inconclusive", "未发现"]):
                negative.append(line)
            elif any(word in lowered for word in ["http", "status", "found", "detected", "confirmed", "json", "header"]):
                confirmed.append(line)
        for item in executed_tasks:
            action_id = item.get("action_id")
            tool = item.get("tool")
            target = item.get("target") or focus.get("target")
            if action_id or tool:
                do_not_repeat.append(f"{tool or 'action'} on {target or 'target'} ({action_id or 'no action_id'})")
        return {
            "confirmed_facts": self._dedupe_trim(confirmed, limits.text)[:12],
            "negative_evidence": self._dedupe_trim(negative, limits.text)[:8],
            "failed_attempts": self._dedupe_trim(failed, limits.text)[:8],
            "missing_evidence": self._dedupe_trim(missing, limits.text)[:8],
            "do_not_repeat": self._dedupe_trim(do_not_repeat, limits.text)[:8],
        }

    def _artifact_refs(
        self,
        *,
        observations: list[dict[str, Any]],
        validation_history: list[dict[str, Any]],
        executed_tasks: list[dict[str, Any]],
        recent_tool_results: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        refs = []
        for item in [*observations, *validation_history, *executed_tasks, *recent_tool_results]:
            artifact_id = item.get("artifact_id")
            if not artifact_id:
                continue
            refs.append(
                {
                    "artifact_id": artifact_id,
                    "action_id": item.get("action_id"),
                    "tool": item.get("tool"),
                    "profile": item.get("profile"),
                    "target": item.get("target"),
                    "summary": self._trim(item.get("summary") or item.get("reasoning") or "", 300),
                }
            )
        return self._dedupe_dicts(refs, key_fields=("artifact_id", "action_id", "tool", "target"))[:limit]

    def _budget_report(self, view: dict[str, Any], *, agent_name: str, level: str) -> dict[str, Any]:
        components = {
            "base_overhead": self.base_overhead_tokens,
            "base_memory_summary": self.estimate_tokens(view.get("base_memory_summary")),
            "current_finding": self.estimate_tokens(view.get("current_finding")),
            "validation_plan": self.estimate_tokens(view.get("validation_plan")),
            "related_observations": self.estimate_tokens(view.get("related_observations")),
            "related_web_context": self.estimate_tokens(view.get("related_web_context")),
            "related_validation_history": self.estimate_tokens(view.get("related_validation_history")),
            "recent_tool_results": self.estimate_tokens(view.get("recent_tool_results")),
            "evidence_ledger": self.estimate_tokens(view.get("evidence_ledger")),
            "artifact_refs": self.estimate_tokens(view.get("artifact_refs")),
        }
        estimated_input = sum(components.values())
        available = max(1, self.context_window_tokens - self.reserved_output_tokens)
        usage_ratio = round(estimated_input / available, 4)
        return {
            "agent": agent_name,
            "context_window": self.context_window_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "estimated_input_tokens": estimated_input,
            "remaining_tokens": max(0, available - estimated_input),
            "usage_ratio": usage_ratio,
            "compression_level": level,
            "components": components,
        }

    def _usage_ratio(self, value: Any) -> float:
        available = max(1, self.context_window_tokens - self.reserved_output_tokens)
        return (self.base_overhead_tokens + self.estimate_tokens(value)) / available

    def estimate_tokens(self, value: Any) -> int:
        text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
        ascii_chars = sum(1 for char in text if ord(char) < 128)
        non_ascii_chars = len(text) - ascii_chars
        return max(1, int(ascii_chars / 4) + int(non_ascii_chars / 2))

    def _compact_finding(self, finding: dict[str, Any], limits: CompressionLimits) -> dict[str, Any]:
        metadata = self._dict_or_empty(finding.get("metadata"))
        return {
            "id": finding.get("id"),
            "title": finding.get("title"),
            "target": finding.get("target"),
            "severity": finding.get("severity"),
            "confidence": finding.get("confidence"),
            "status": finding.get("status"),
            "category": metadata.get("category"),
            "description": self._trim(finding.get("description", ""), limits.text),
            "evidence": [self._trim(item, limits.text) for item in finding.get("evidence", [])[:8]],
            "recommendation": self._trim(finding.get("recommendation", ""), limits.text),
        }

    def _compact_validation_plan(self, plan: dict[str, Any], limits: CompressionLimits) -> dict[str, Any]:
        if not plan:
            return {}
        return {
            "id": plan.get("id"),
            "finding_id": plan.get("finding_id"),
            "target": plan.get("target"),
            "objective": self._trim(plan.get("objective", ""), limits.text),
            "risk_level": plan.get("risk_level"),
            "success_criteria": [self._trim(item, limits.text) for item in plan.get("success_criteria", [])[:8]],
            "failure_criteria": [self._trim(item, limits.text) for item in plan.get("failure_criteria", [])[:8]],
            "actions": [
                {
                    "name": item.get("name"),
                    "tool": item.get("tool"),
                    "profile": item.get("profile"),
                    "target": item.get("target"),
                    "risk_level": item.get("risk_level"),
                    "rationale": self._trim(item.get("rationale", ""), limits.text),
                }
                for item in plan.get("actions", [])[:5]
                if isinstance(item, dict)
            ],
        }

    def _compact_observation(self, item: dict[str, Any], limits: CompressionLimits) -> dict[str, Any]:
        return {
            "id": item.get("id"),
            "action_id": item.get("action_id"),
            "tool": item.get("tool"),
            "profile": item.get("profile"),
            "target": item.get("target"),
            "status": item.get("status"),
            "risk_level": item.get("risk_level"),
            "summary": self._trim(item.get("summary", ""), limits.text),
            "artifact_id": item.get("artifact_id"),
            "signals": [
                {
                    "kind": signal.get("kind"),
                    "name": signal.get("name"),
                    "severity": signal.get("severity"),
                    "target": signal.get("target"),
                    "evidence": self._trim(signal.get("evidence", ""), limits.text),
                }
                for signal in item.get("signals", [])[: limits.signals]
                if isinstance(signal, dict)
            ],
        }

    def _compact_web_context(self, item: dict[str, Any], limits: CompressionLimits) -> dict[str, Any]:
        return {
            "target": item.get("target"),
            "status_code": item.get("status_code"),
            "title": self._trim(item.get("title", ""), 250),
            "links": item.get("links", [])[:20],
            "interesting_paths": item.get("interesting_paths", [])[:20],
            "forms": [
                {
                    "action": form.get("action"),
                    "method": form.get("method"),
                    "inputs": len(form.get("inputs", [])),
                }
                for form in item.get("forms", [])[:6]
                if isinstance(form, dict)
            ],
            "error": self._trim(item.get("error", ""), limits.text),
        }

    def _compact_validation_history(self, item: dict[str, Any], limits: CompressionLimits) -> dict[str, Any]:
        metadata = self._dict_or_empty(item.get("metadata"))
        return {
            "id": item.get("id"),
            "finding_id": item.get("finding_id"),
            "validation_plan_id": item.get("validation_plan_id"),
            "status": item.get("status") or item.get("decision"),
            "confidence": item.get("confidence"),
            "target": item.get("target"),
            "reasoning": self._trim(item.get("reasoning", ""), limits.text),
            "impact": self._trim(item.get("impact", ""), limits.text),
            "evidence": [self._trim(value, limits.text) for value in item.get("evidence", [])[:8]],
            "missing_evidence": metadata.get("react_missing_evidence", [])[:8],
            "artifact_id": item.get("artifact_id"),
        }

    def _compact_tool_result(self, item: dict[str, Any], limits: CompressionLimits) -> dict[str, Any]:
        return {
            "tool": item.get("tool") or item.get("name"),
            "profile": item.get("profile"),
            "target": item.get("target"),
            "status": item.get("status"),
            "summary": self._trim(item.get("summary", ""), limits.text),
            "error": self._trim(item.get("error", ""), limits.text),
            "stdout": self._trim(item.get("stdout", ""), limits.text),
            "stderr": self._trim(item.get("stderr", ""), limits.text // 2),
            "action_id": item.get("action_id"),
            "artifact_id": item.get("artifact_id"),
        }

    def _compact_executed_task(self, item: dict[str, Any], limits: CompressionLimits) -> dict[str, Any]:
        task = self._dict_or_empty(item.get("task"))
        return {
            "action_id": item.get("action_id"),
            "status": item.get("status"),
            "tool": item.get("tool") or task.get("tool"),
            "profile": item.get("profile") or task.get("profile"),
            "target": item.get("target") or task.get("target"),
            "summary": self._trim(item.get("summary", ""), limits.text),
            "error": self._trim(item.get("error", ""), limits.text),
            "artifact_id": item.get("artifact_id"),
        }

    def _fallback_view(
        self,
        state: AutoFlowState,
        agent_name: str,
        focus: dict[str, Any],
        base_memory: dict[str, Any],
    ) -> dict[str, Any]:
        view = {
            "view_type": agent_name,
            "focus": focus,
            "base_memory_summary": {
                "project_id": base_memory.get("project_id") or state.get("project_id"),
                "flow_id": base_memory.get("flow_id") or state.get("flow_id"),
                "memory_counts": base_memory.get("memory_counts", {}),
            },
        }
        view["budget_report"] = self._budget_report(view, agent_name=agent_name, level="level_0")
        return view

    def _observation_lines(self, item: dict[str, Any]) -> list[str]:
        lines = []
        if item.get("summary"):
            lines.append(f"{item.get('tool')}/{item.get('profile')}: {item.get('summary')}")
        for signal in item.get("signals", []):
            evidence = signal.get("evidence") or signal.get("name") or ""
            if evidence:
                lines.append(f"{signal.get('kind')}: {evidence}")
        return lines

    def _history_lines(self, item: dict[str, Any]) -> list[str]:
        lines = []
        for key in ("status", "reasoning", "impact"):
            if item.get(key):
                lines.append(f"{key}: {item.get(key)}")
        lines.extend(str(value) for value in item.get("evidence", []) if value)
        lines.extend(f"missing_evidence: {value}" for value in item.get("missing_evidence", []) if value)
        return lines

    def _task_lines(self, item: dict[str, Any]) -> list[str]:
        lines = []
        if item.get("summary"):
            lines.append(f"{item.get('tool')}/{item.get('profile')}: {item.get('summary')}")
        if item.get("error"):
            lines.append(f"failed: {item.get('error')}")
        return lines

    def _tool_result_lines(self, item: dict[str, Any]) -> list[str]:
        return [
            str(value)
            for value in [item.get("summary"), item.get("error"), item.get("stdout"), item.get("stderr")]
            if value
        ]

    def _take_ranked(self, ranked: list[tuple[int, dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
        return [item for _, item in ranked[:limit]]

    def _item_text(self, item: dict[str, Any]) -> str:
        return json.dumps(item, ensure_ascii=False, default=str).lower()

    def _dict_list(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    def _dict_or_empty(self, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _dedupe_dicts(self, items: list[dict[str, Any]], *, key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
        seen = set()
        result = []
        for item in items:
            key = tuple(item.get(field) for field in key_fields if item.get(field) not in (None, ""))
            if not key:
                key = (json.dumps(item, sort_keys=True, default=str),)
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _dedupe_trim(self, values: list[str], limit: int) -> list[str]:
        seen = set()
        result = []
        for value in values:
            text = self._trim(value, limit)
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def _trim(self, value: object, limit: int) -> str:
        text = str(value or "")
        return text if len(text) <= limit else text[: limit - 3] + "..."
