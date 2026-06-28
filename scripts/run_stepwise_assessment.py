"""Run AutoFlow stage by stage and print progress after every agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoflow.agents.discovery import DiscoveryAgent
from autoflow.agents.discovery_reasoner import DiscoveryReasonerAgent
from autoflow.agents.executor import ExecutorAgent, ValidationExecutorAgent
from autoflow.agents.planner import PlannerAgent
from autoflow.agents.recon import ReconAgent
from autoflow.agents.reporter import ReporterAgent
from autoflow.agents.validation import ValidationAgent
from autoflow.agents.verifier import VerifierAgent
from autoflow.artifacts.store import ArtifactStore
from autoflow.memory.redis_store import RedisMemoryStore
from autoflow.settings import settings


DEFAULT_PROMPT = (
    "这是用户授权的 Web 靶场。请进行分阶段 Web 渗透测试，先完成低风险信息收集、"
    "网站结构识别、入口点发现和常见配置验证；中高风险动作需要审批后执行，不进行破坏性操作。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an AutoFlow assessment step by step.")
    parser.add_argument("--target", required=True, help="Authorized target, for example 192.168.34.191:3001.")
    parser.add_argument("--project", default="autoflow-stepwise", help="Project name.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User assessment prompt.")
    parser.add_argument("--output", help="Optional Markdown report output path.")
    parser.add_argument("--offline-planner", action="store_true", help="Disable LLM planning/reasoning agents for offline tests.")
    parser.add_argument("--max-rounds", type=int, default=3, help="Maximum strategy rounds as a loop guard.")
    parser.add_argument(
        "--execute-limit",
        type=int,
        default=0,
        help="Limit executable auto actions before Executor. 0 means no limit.",
    )
    parser.add_argument(
        "--validation-execute-limit",
        type=int,
        default=None,
        help="Limit executable validation actions before ValidationExecutor. Defaults to --execute-limit. 0 means no limit.",
    )
    return parser.parse_args()


def emit(stage: str, payload: dict) -> None:
    print(f"\n=== {stage} ===", flush=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def record_redis(stage: str, state: dict, store: RedisMemoryStore) -> None:
    error = store.record_node_state(stage.lower(), state)
    if error:
        state["redis_memory_error"] = error


def summarize_state(state: dict) -> dict:
    flow = state.get("flow")
    return {
        "current_phase": state.get("current_phase"),
        "next_action": state.get("next_action"),
        "flow_id": flow.id if flow else state.get("flow_id"),
        "task_count": len(flow.tasks) if flow else 0,
        "assets": len(state.get("assets", [])),
        "web_recon": len(state.get("web_recon", [])),
        "attack_surfaces": len(state.get("attack_surfaces", [])),
        "test_plans": len(state.get("test_plans", [])),
        "executed_tasks": len(state.get("executed_tasks", [])),
        "tool_observations": len(state.get("tool_observations", [])),
        "findings": len(state.get("findings", [])),
        "validation_plans": len(state.get("validation_plans", [])),
        "approvals_required": len(state.get("approvals_required", [])),
    }


def limit_auto_actions(state: dict, limit: int) -> None:
    if limit <= 0:
        return
    remaining = limit
    filtered_plans = []
    for plan in state.get("test_plans", []):
        actions = []
        for action in plan.get("actions", []):
            if action.get("requires_approval") or action.get("risk_level") != "low":
                actions.append(action)
                continue
            if remaining > 0:
                actions.append(action)
                remaining -= 1
        filtered_plan = {**plan, "actions": actions}
        filtered_plans.append(filtered_plan)
    state["test_plans"] = filtered_plans


def limit_validation_actions(state: dict, limit: int) -> None:
    if limit <= 0:
        state.pop("validation_action_budget", None)
        state.pop("validation_react_finding_budget", None)
        return
    state["validation_action_budget"] = limit
    state["validation_react_finding_budget"] = limit


async def run() -> dict:
    args = parse_args()
    validation_execute_limit = args.execute_limit if args.validation_execute_limit is None else args.validation_execute_limit
    if not settings.llm_api_key and not args.offline_planner:
        raise SystemExit("LLM_API_KEY is required. Use --offline-planner to skip Planner LLM.")

    artifact_store = ArtifactStore()
    state = {
        "project_id": args.project,
        "target_scope": [args.target],
        "user_prompt": args.prompt,
        "rules_of_engagement": {
            "authorized": True,
            "allow_exploit": False,
            "allow_bruteforce": False,
            "require_approval_for": ["medium", "high", "critical"],
        },
        "max_rounds": args.max_rounds,
    }

    planner = PlannerAgent(use_llm=False) if args.offline_planner else PlannerAgent()
    recon = ReconAgent(artifact_store=artifact_store)
    discovery_reasoner = DiscoveryReasonerAgent(use_llm=False) if args.offline_planner else DiscoveryReasonerAgent()
    discovery = DiscoveryAgent(recon=recon, reasoner=discovery_reasoner)
    executor = ExecutorAgent(artifact_store=artifact_store)
    validation_executor = ValidationExecutorAgent(artifact_store=artifact_store)
    verifier = VerifierAgent()
    validation = ValidationAgent()
    reporter = ReporterAgent()
    redis_store = RedisMemoryStore.from_settings()

    emit("START", {"target": args.target, "project": args.project, "prompt": args.prompt})

    state = await planner.run(state)
    record_redis("planner", state, redis_store)
    flow = state["flow"]
    if args.prompt:
        flow.metadata["user_prompt"] = args.prompt
    emit(
        "PLANNER",
        {
            **summarize_state(state),
            "tasks": [
                {
                    "id": task.id,
                    "type": task.type,
                    "target": task.target,
                    "risk": task.risk_level.value,
                    "objective": task.objective,
                }
                for task in flow.tasks
            ],
        },
    )

    state = await discovery.run(state)
    record_redis("discovery", state, redis_store)
    emit(
        "DISCOVERY",
        {
            **summarize_state(state),
            "assets": state.get("assets", []),
            "web_recon": [
                {
                    "target": item.get("target"),
                    "status_code": item.get("status_code"),
                    "title": item.get("title"),
                    "links": len(item.get("links", [])),
                    "forms": len(item.get("forms", [])),
                    "scripts": len(item.get("scripts", [])),
                    "interesting_paths": len(item.get("interesting_paths", [])),
                    "error": item.get("error", ""),
                }
                for item in state.get("web_recon", [])
            ],
            "attack_surfaces": [
                {
                    "target": item.get("target"),
                    "surface_type": item.get("surface_type"),
                    "entrypoints": len(item.get("entrypoints", [])),
                    "technology": item.get("technology"),
                }
                for item in state.get("attack_surfaces", [])
            ],
            "memory_context_has_web_recon": bool(state.get("memory_context", {}).get("web_recon")),
            "test_plans": [
                {
                    "id": plan.get("id"),
                    "strategy": plan.get("strategy"),
                    "target": plan.get("target"),
                    "risk": plan.get("risk_level"),
                    "actions": [
                        {
                            "id": action.get("id"),
                            "tool": action.get("tool"),
                            "profile": action.get("profile"),
                            "risk": action.get("risk_level"),
                            "approval": action.get("requires_approval"),
                        }
                        for action in plan.get("actions", [])
                    ],
                }
                for plan in state.get("test_plans", [])
            ],
        },
    )

    limit_auto_actions(state, args.execute_limit)
    if args.execute_limit:
        emit("EXECUTE_LIMIT", {"execute_limit": args.execute_limit})

    state = await executor.run(state)
    record_redis("executor", state, redis_store)
    emit(
        "EXECUTOR",
        {
            **summarize_state(state),
            "executed_tasks": [
                {
                    "action_id": item.get("action_id"),
                    "status": item.get("status"),
                    "tool": item.get("task", {}).get("tool"),
                    "profile": item.get("task", {}).get("profile"),
                    "summary": item.get("summary", "")[:300],
                    "error": item.get("error", "")[:300],
                }
                for item in state.get("executed_tasks", [])
            ],
            "approvals_required": [
                {
                    "action_id": item.get("action_id"),
                    "tool": item.get("tool"),
                    "profile": item.get("profile"),
                    "risk": item.get("risk_level"),
                }
                for item in state.get("approvals_required", [])
            ],
            "tool_observations": [
                {
                    "tool": item.get("tool"),
                    "profile": item.get("profile"),
                    "status": item.get("status"),
                    "signals": [
                        {
                            "kind": signal.get("kind"),
                            "name": signal.get("name"),
                            "severity": signal.get("severity"),
                        }
                        for signal in item.get("signals", [])[:5]
                    ],
                }
                for item in state.get("tool_observations", [])
            ],
        },
    )

    state = await verifier.run(state)
    record_redis("verifier", state, redis_store)
    emit(
        "VERIFIER",
        {
            **summarize_state(state),
            "verification": state.get("verification", {}),
            "findings": [
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "target": item.get("target"),
                    "severity": item.get("severity"),
                    "confidence": item.get("confidence"),
                    "status": item.get("status"),
                }
                for item in state.get("findings", [])
            ],
        },
    )

    state = await validation.run(state)
    record_redis("validation", state, redis_store)
    emit(
        "VALIDATION",
        {
            **summarize_state(state),
            "validation_plans": [
                {
                    "id": plan.get("id"),
                    "finding_id": plan.get("finding_id"),
                    "target": plan.get("target"),
                    "objective": plan.get("objective"),
                    "risk": plan.get("risk_level"),
                    "approval": plan.get("requires_approval"),
                    "actions": [
                        {
                            "id": action.get("id"),
                            "tool": action.get("tool"),
                            "profile": action.get("profile"),
                            "kind": action.get("action_kind"),
                            "risk": action.get("risk_level"),
                            "approval": action.get("requires_approval"),
                        }
                        for action in plan.get("actions", [])
                    ],
                }
                for plan in state.get("validation_plans", [])
            ],
        },
    )

    if state.get("next_action") == "validation_execute":
        limit_validation_actions(state, validation_execute_limit)
        if validation_execute_limit:
            emit("VALIDATION_EXECUTE_LIMIT", {"validation_execute_limit": validation_execute_limit})
        state = await validation_executor.run(state)
        record_redis("validation_executor", state, redis_store)
        emit(
            "VALIDATION_EXECUTOR",
            {
                **summarize_state(state),
                "executed_tasks": [
                    {
                        "action_id": item.get("action_id"),
                        "status": item.get("status"),
                        "tool": item.get("task", {}).get("tool"),
                        "profile": item.get("task", {}).get("profile"),
                        "summary": item.get("summary", "")[:300],
                        "error": item.get("error", "")[:300],
                    }
                    for item in state.get("executed_tasks", [])
                ],
                "validation_plans": [
                    {
                        "id": plan.get("id"),
                        "finding_id": plan.get("finding_id"),
                        "target": plan.get("target"),
                        "status": plan.get("status"),
                        "risk": plan.get("risk_level"),
                        "results": len(plan.get("execution_results", [])),
                    }
                    for plan in state.get("validation_plans", [])
                ],
                "tool_observations": len(state.get("tool_observations", [])),
            },
        )

    while state.get("next_action") == "strategy":
        state = await discovery_reasoner.run(state)
        record_redis(f"strategy_round_{state.get('strategy_round')}", state, redis_store)
        emit(
            f"STRATEGY_ROUND_{state.get('strategy_round')}",
            {
                **summarize_state(state),
                "strategy_round": state.get("strategy_round"),
                "max_rounds": state.get("max_rounds", 3),
                "test_plans": [
                    {
                        "id": plan.get("id"),
                        "strategy": plan.get("strategy"),
                        "target": plan.get("target"),
                        "risk": plan.get("risk_level"),
                        "actions": [
                            {
                                "id": action.get("id"),
                                "tool": action.get("tool"),
                                "profile": action.get("profile"),
                                "kind": action.get("action_kind"),
                                "risk": action.get("risk_level"),
                                "approval": action.get("requires_approval"),
                            }
                            for action in plan.get("actions", [])
                        ],
                    }
                    for plan in state.get("test_plans", [])
                ],
            },
        )
        if state.get("next_action") != "execute":
            break

        state = await executor.run(state)
        record_redis(f"executor_round_{state.get('strategy_round')}", state, redis_store)
        emit(
            f"EXECUTOR_ROUND_{state.get('strategy_round')}",
            {
                **summarize_state(state),
                "executed_tasks": [
                    {
                        "action_id": item.get("action_id"),
                        "status": item.get("status"),
                        "tool": item.get("task", {}).get("tool"),
                        "profile": item.get("task", {}).get("profile"),
                        "summary": item.get("summary", "")[:300],
                        "error": item.get("error", "")[:300],
                    }
                    for item in state.get("executed_tasks", [])
                ],
                "approvals_required": [
                    {
                        "action_id": item.get("action_id"),
                        "tool": item.get("tool"),
                        "profile": item.get("profile"),
                        "risk": item.get("risk_level"),
                    }
                    for item in state.get("approvals_required", [])
                ],
            },
        )

        state = await verifier.run(state)
        record_redis(f"verifier_round_{state.get('strategy_round')}", state, redis_store)
        emit(
            f"VERIFIER_ROUND_{state.get('strategy_round')}",
            {
                **summarize_state(state),
                "verification": state.get("verification", {}),
                "findings": [
                    {
                        "title": item.get("title"),
                        "target": item.get("target"),
                        "severity": item.get("severity"),
                        "confidence": item.get("confidence"),
                    }
                    for item in state.get("findings", [])
                ],
            },
        )

        state = await validation.run(state)
        record_redis(f"validation_round_{state.get('strategy_round')}", state, redis_store)
        emit(
            f"VALIDATION_ROUND_{state.get('strategy_round')}",
            {
                **summarize_state(state),
                "validation_plans": [
                    {
                        "id": plan.get("id"),
                        "finding_id": plan.get("finding_id"),
                        "target": plan.get("target"),
                        "objective": plan.get("objective"),
                        "risk": plan.get("risk_level"),
                        "approval": plan.get("requires_approval"),
                    }
                    for plan in state.get("validation_plans", [])
                ],
            },
        )

        if state.get("next_action") == "validation_execute":
            limit_validation_actions(state, validation_execute_limit)
            if validation_execute_limit:
                emit(
                    f"VALIDATION_EXECUTE_LIMIT_ROUND_{state.get('strategy_round')}",
                    {"validation_execute_limit": validation_execute_limit},
                )
            state = await validation_executor.run(state)
            record_redis(f"validation_executor_round_{state.get('strategy_round')}", state, redis_store)
            emit(
                f"VALIDATION_EXECUTOR_ROUND_{state.get('strategy_round')}",
                {
                    **summarize_state(state),
                    "executed_tasks": [
                        {
                            "action_id": item.get("action_id"),
                            "status": item.get("status"),
                            "tool": item.get("task", {}).get("tool"),
                            "profile": item.get("task", {}).get("profile"),
                            "summary": item.get("summary", "")[:300],
                            "error": item.get("error", "")[:300],
                        }
                        for item in state.get("executed_tasks", [])
                    ],
                    "validation_plans": [
                        {
                            "id": plan.get("id"),
                            "finding_id": plan.get("finding_id"),
                            "target": plan.get("target"),
                            "status": plan.get("status"),
                            "risk": plan.get("risk_level"),
                            "results": len(plan.get("execution_results", [])),
                        }
                        for plan in state.get("validation_plans", [])
                    ],
                    "tool_observations": len(state.get("tool_observations", [])),
                },
            )

    state = await reporter.run(state)
    record_redis("reporter", state, redis_store)
    emit(
        "REPORTER",
        {
            **summarize_state(state),
            "report_length": len(state.get("report_markdown", "")),
            "report_has_web_recon": "## Web Recon" in state.get("report_markdown", ""),
        },
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(state.get("report_markdown", ""), encoding="utf-8")
        emit("OUTPUT", {"report": str(output_path)})

    return state


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
