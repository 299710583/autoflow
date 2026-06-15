from __future__ import annotations

from autoflow.agents.base import BaseAgent
from autoflow.agents.tool_loop import AgentToolLoop
from autoflow.flows.models import FlowStatus, MemoryItem, MemoryKind
from autoflow.graph.state import AutoFlowState
from autoflow.settings import settings


REPORTER_SYSTEM_PROMPT = """You are AutoFlow's ReporterAgent for an authorized security assessment.
Use memory and observation tools when helpful to summarize evidence.
Return only JSON. Do not include markdown.
Do not call active scanning or validation tools during reporting unless explicitly necessary; prefer memory and observation search tools.
"""


class ReporterAgent(BaseAgent):
    """根据资产和已执行任务生成当前 Markdown 报告。"""

    name = "reporter"

    def __init__(
        self,
        tool_loop: AgentToolLoop | None = None,
        use_tool_calling: bool | None = None,
    ) -> None:
        self.tool_loop = tool_loop
        self.use_tool_calling = use_tool_calling

    async def run(self, state: AutoFlowState) -> AutoFlowState:
        state["current_phase"] = "reporting"
        flow = state.get("flow")
        assets = state.get("assets", [])
        executed_tasks = state.get("executed_tasks", [])
        tool_observations = state.get("tool_observations", [])
        findings = state.get("findings", [])
        validation_plans = state.get("validation_plans", [])
        validation_results = state.get("validation_results", [])
        test_plans = state.get("test_plans", [])
        web_recon = state.get("web_recon", [])
        verification = state.get("verification", {})
        report_notes = self._build_report_notes_with_tool_loop(state) if self._should_use_tool_loop() else {}
        report = self._build_markdown_report(
            flow_name=flow.name if flow else "AutoFlow",
            assets=assets,
            web_recon=web_recon,
            executed_tasks=executed_tasks,
            tool_observations=tool_observations,
            findings=findings,
            validation_plans=validation_plans,
            validation_results=validation_results,
            test_plans=test_plans,
            report_notes=report_notes,
        )

        # 报告先保存在图状态中，CLI 调用方也可以写入磁盘。
        state["report_markdown"] = report
        state["next_action"] = "end"

        if flow is not None:
            flow.status = FlowStatus.COMPLETED
            flow.add_memory(
                MemoryItem(
                    kind=MemoryKind.LESSON,
                    content=verification.get("summary", "Generated assessment report"),
                    source=self.name,
                )
            )

        return state

    def _should_use_tool_loop(self) -> bool:
        if self.use_tool_calling is not None:
            return self.use_tool_calling
        return bool(settings.llm_api_key)

    def _build_report_notes_with_tool_loop(self, state: AutoFlowState) -> dict:
        loop = self.tool_loop or AgentToolLoop(max_tool_rounds=3, max_tool_calls=4)
        payload = {
            "task": "Create concise report notes from existing AutoFlow state. Prefer memory/observation tools.",
            "counts": {
                "assets": len(state.get("assets", [])),
                "executed_tasks": len(state.get("executed_tasks", [])),
                "tool_observations": len(state.get("tool_observations", [])),
                "findings": len(state.get("findings", [])),
                "validation_plans": len(state.get("validation_plans", [])),
                "validation_results": len(state.get("validation_results", [])),
            },
            "findings": state.get("findings", [])[:40],
            "validation_plans": state.get("validation_plans", [])[:40],
            "validation_results": state.get("validation_results", [])[:40],
            "final_output_schema": {
                "report_notes": {
                    "summary": "short assessment summary",
                    "validated_highlights": ["important validated or strongly supported points"],
                    "remaining_work": ["important next steps"],
                }
            },
        }
        try:
            tools = None
            if hasattr(loop, "catalog"):
                tools = [
                    tool
                    for tool in loop.catalog.openai_tools()
                    if tool["function"]["name"] in {"read_agent_memory", "search_observations", "list_known_targets"}
                ]
            result = loop.run(
                system_prompt=REPORTER_SYSTEM_PROMPT,
                user_payload=payload,
                state=state,
                final_repair_instruction="Return a JSON object with report_notes.",
                tools=tools,
            )
        except Exception:
            return {}
        state["reporter_tool_loop_messages"] = result.messages
        state["reporter_tool_loop_results"] = result.tool_results
        notes = result.final.get("report_notes", {})
        return notes if isinstance(notes, dict) else {}

    def _build_markdown_report(
        self,
        flow_name: str,
        assets: list[dict],
        web_recon: list[dict] | None = None,
        executed_tasks: list[dict] | None = None,
        tool_observations: list[dict] | None = None,
        findings: list[dict] | None = None,
        validation_plans: list[dict] | None = None,
        validation_results: list[dict] | None = None,
        test_plans: list[dict] | None = None,
        report_notes: dict | None = None,
    ) -> str:
        lines = [
            f"# {flow_name} Report",
            "",
            "## Assets",
            "",
        ]
        if not assets:
            lines.append("No assets discovered.")
            return "\n".join(lines)

        report_notes = report_notes or {}
        if report_notes:
            lines.extend(["", "## LLM Report Notes", ""])
            if report_notes.get("summary"):
                lines.append(str(report_notes["summary"]))
                lines.append("")
            for label, key in [
                ("Validated highlights", "validated_highlights"),
                ("Remaining work", "remaining_work"),
            ]:
                values = report_notes.get(key, [])
                if isinstance(values, list) and values:
                    lines.append(f"- {label}:")
                    for value in values[:10]:
                        lines.append(f"  - {value}")
            lines.append("")

        for asset in assets:
            lines.append(f"### {asset.get('ip', 'unknown')}")
            hostname = asset.get("hostname")
            if hostname:
                lines.append(f"- Hostname: {hostname}")
            ports = asset.get("ports", [])
            if not ports:
                lines.append("- Open ports: none")
                continue
            lines.append("- Open ports:")
            for port in ports:
                service = port.get("service") or "unknown"
                product = port.get("product") or ""
                version = port.get("version") or ""
                detail = " ".join(value for value in [service, product, version] if value)
                lines.append(f"  - {port.get('protocol', 'tcp')}/{port.get('port')}: {detail}")
            lines.append("")

        web_recon = web_recon or []
        if web_recon:
            lines.extend(["", "## Web Recon", ""])
            for item in web_recon:
                lines.append(f"### {item.get('target', 'unknown')}")
                lines.append(f"- Status: {item.get('status_code', 0)}")
                if item.get("title"):
                    lines.append(f"- Title: {item.get('title')}")
                lines.append(f"- Links discovered: {len(item.get('links', []))}")
                lines.append(f"- Forms discovered: {len(item.get('forms', []))}")
                lines.append(f"- Scripts discovered: {len(item.get('scripts', []))}")
                if item.get("forms"):
                    lines.append("- Forms:")
                    for form in item.get("forms", [])[:10]:
                        inputs = ", ".join(
                            value.get("name") or value.get("id") or value.get("type", "input")
                            for value in form.get("inputs", [])
                        )
                        lines.append(
                            f"  - {form.get('method', 'get').upper()} {form.get('action', '')} "
                            f"inputs=[{inputs}]"
                        )
                if item.get("interesting_paths"):
                    lines.append("- Interesting paths:")
                    for path in item.get("interesting_paths", [])[:20]:
                        lines.append(f"  - {path}")
                lines.append("")

        executed_tasks = executed_tasks or []
        if executed_tasks:
            # 展示后续工具执行结果，避免报告只停留在 recon 阶段。
            lines.extend(["", "## Follow-Up Tasks", ""])
            for item in executed_tasks:
                task = item.get("task", {})
                lines.append(f"- {task.get('type', 'task')} on {task.get('target', 'unknown')}: {item.get('status')}")
                if item.get("summary"):
                    lines.append(f"  - Summary: {item['summary']}")
                if item.get("error"):
                    lines.append(f"  - Error: {item['error']}")

        tool_observations = tool_observations or []
        if tool_observations:
            lines.extend(["", "## Tool Observations", ""])
            for observation in tool_observations:
                lines.append(
                    f"### {observation.get('tool', 'tool')}/{observation.get('profile', '')} "
                    f"on {observation.get('target', 'unknown')}"
                )
                lines.append(f"- Status: {observation.get('status', 'unknown')}")
                lines.append(f"- Signals: {len(observation.get('signals', []))}")
                for signal in observation.get("signals", [])[:10]:
                    lines.append(
                        "  - "
                        f"{signal.get('kind', 'signal')}: {signal.get('name', '')} "
                        f"severity={signal.get('severity', 'info')}"
                    )
                if observation.get("summary"):
                    lines.append(f"- Summary: {observation.get('summary')}")
                lines.append("")

        findings = findings or []
        if findings:
            lines.extend(["", "## Findings", ""])
            for finding in findings:
                lines.append(f"### {finding.get('title', 'Untitled finding')}")
                lines.append(f"- Status: {finding.get('status', 'candidate')}")
                lines.append(f"- Severity: {finding.get('severity', 'info')}")
                lines.append(f"- Confidence: {finding.get('confidence', 'medium')}")
                lines.append(f"- Target: {finding.get('target', 'unknown')}")
                lines.append(f"- Description: {finding.get('description', '')}")
                evidence = finding.get("evidence", [])
                if evidence:
                    lines.append("- Evidence:")
                    for item in evidence:
                        lines.append(f"  - {item}")
                recommendation = finding.get("recommendation")
                if recommendation:
                    lines.append(f"- Recommendation: {recommendation}")
                lines.append("")

        validation_results = validation_results or []
        if validation_results:
            lines.extend(["", "## Validation Results", ""])
            for result in validation_results:
                lines.append(f"### {result.get('finding_id', 'finding')} via {result.get('validation_plan_id', '')}")
                lines.append(f"- Status: {result.get('status', 'inconclusive')}")
                lines.append(f"- Confidence: {result.get('confidence', 'medium')}")
                if result.get("impact"):
                    lines.append(f"- Impact: {result.get('impact')}")
                if result.get("reasoning"):
                    lines.append(f"- Reasoning: {result.get('reasoning')}")
                evidence = result.get("evidence", [])
                if evidence:
                    lines.append("- Validation evidence:")
                    for item in evidence[:10]:
                        lines.append(f"  - {item}")
                steps = result.get("reproduction_steps", [])
                if steps:
                    lines.append("- Reproduction steps:")
                    for step in steps[:10]:
                        lines.append(f"  - {step}")
                lines.append("")

        validation_plans = validation_plans or []
        if validation_plans:
            lines.extend(["", "## Validation Plans", ""])
            for plan in validation_plans:
                lines.append(f"### {plan.get('objective', 'Validation plan')}")
                lines.append(f"- Finding ID: {plan.get('finding_id', '')}")
                lines.append(f"- Target: {plan.get('target', 'unknown')}")
                lines.append(f"- Status: {plan.get('status', 'planned')}")
                lines.append(f"- Risk: {plan.get('risk_level', 'medium')}")
                lines.append(f"- Requires approval: {plan.get('requires_approval', True)}")
                rationale = plan.get("rationale")
                if rationale:
                    lines.append(f"- Rationale: {rationale}")
                success = plan.get("success_criteria", [])
                if success:
                    lines.append("- Success criteria:")
                    for item in success:
                        lines.append(f"  - {item}")
                failure = plan.get("failure_criteria", [])
                if failure:
                    lines.append("- Failure criteria:")
                    for item in failure:
                        lines.append(f"  - {item}")
                actions = plan.get("actions", [])
                if actions:
                    lines.append("- Validation actions:")
                    for action in actions:
                        lines.append(
                            "  - "
                            f"{action.get('name', 'action')} "
                            f"({action.get('tool', '')}/{action.get('profile', '')}, "
                            f"risk={action.get('risk_level', 'low')}, "
                            f"approval={action.get('requires_approval', False)})"
                        )
                execution_results = plan.get("execution_results", [])
                if execution_results:
                    lines.append("- Execution results:")
                    for result in execution_results:
                        lines.append(
                            "  - "
                            f"{result.get('action_id', '')}: {result.get('status', 'unknown')}"
                        )
                        if result.get("summary"):
                            lines.append(f"    - Summary: {result.get('summary')}")
                        if result.get("error"):
                            lines.append(f"    - Error: {result.get('error')}")
                lines.append("")

        test_plans = test_plans or []
        if test_plans:
            lines.extend(["", "## Test Plans", ""])
            for plan in test_plans:
                lines.append(f"### {plan.get('strategy', 'test_plan')} on {plan.get('target', 'unknown')}")
                lines.append(f"- Angle: {plan.get('angle', '')}")
                lines.append(f"- Risk: {plan.get('risk_level', 'low')}")
                lines.append(f"- Requires approval: {plan.get('requires_approval', False)}")
                rationale = plan.get("rationale")
                if rationale:
                    lines.append(f"- Rationale: {rationale}")
                actions = plan.get("actions", [])
                if actions:
                    lines.append("- Candidate actions:")
                    for action in actions:
                        lines.append(
                            "  - "
                            f"{action.get('name', 'action')} "
                            f"({action.get('tool', '')}/{action.get('profile', '')}, "
                            f"risk={action.get('risk_level', 'low')}, "
                            f"approval={action.get('requires_approval', False)})"
                        )
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"
