"""Run an AutoFlow assessment from the command line."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoflow.agents.planner import PlannerAgent
from autoflow.agents.recon import ReconAgent
from autoflow.agents.discovery_reasoner import DiscoveryReasonerAgent
from autoflow.artifacts.store import ArtifactStore
from autoflow.executor.command_builder import CommandBuilder
from autoflow.executor.ssh_executor import CommandResult
from autoflow.graph.builder import build_assessment_graph
from autoflow.graph.checkpoints import build_async_configured_checkpointer, checkpoint_config
from autoflow.settings import settings


class DryRunKaliClient:
    def __init__(self) -> None:
        self.builder = CommandBuilder()

    def build_command(self, intent: dict):
        return self.builder.build(intent)

    def execute_spec(self, spec):
        output_path = Path(spec.command[-2])
        target = spec.command[-1]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_fake_nmap_xml(target), encoding="utf-8")
        return CommandResult(
            command=spec.command,
            command_text=" ".join(spec.command),
            exit_code=0,
            stdout="dry-run nmap output generated\n",
            stderr="",
        )


def _fake_nmap_xml(target: str) -> str:
    return f"""<?xml version="1.0"?>
<nmaprun>
  <host>
    <status state="up"/>
    <address addr="{target}" addrtype="ipv4"/>
    <ports>
      <port protocol="tcp" portid="22">
        <state state="open"/>
        <service name="ssh" product="OpenSSH" version="9.2"/>
      </port>
      <port protocol="tcp" portid="80">
        <state state="open"/>
        <service name="http" product="nginx"/>
      </port>
    </ports>
  </host>
</nmaprun>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an AutoFlow assessment.")
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        help="Authorized target to assess. Repeat this option for multiple targets.",
    )
    parser.add_argument(
        "--project",
        default="autoflow-assessment",
        help="Assessment project name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use fake Kali execution while still running the LLM planner.",
    )
    parser.add_argument(
        "--offline-agents",
        action="store_true",
        help="Use rule-based Planner/Researcher while still executing real tools unless --dry-run is set.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the generated Markdown report.",
    )
    parser.add_argument(
        "--checkpoint-backend",
        choices=["auto", "none", "memory", "redis"],
        default=settings.checkpoint_backend,
        help="LangGraph checkpoint backend. auto uses Redis when REDIS_ENABLED=true.",
    )
    parser.add_argument(
        "--thread-id",
        help="LangGraph checkpoint thread id. Defaults to project name when a checkpointer is enabled.",
    )
    return parser.parse_args()


async def run() -> dict:
    args = parse_args()
    if not settings.llm_api_key and not args.offline_agents:
        raise SystemExit("LLM_API_KEY is required. Add it to .env or the process environment.")

    artifact_store = ArtifactStore()
    planner = PlannerAgent(use_llm=False) if args.offline_agents else None
    discovery_reasoner = DiscoveryReasonerAgent(use_llm=False) if args.offline_agents else None
    recon = None
    if args.dry_run:
        recon = ReconAgent(kali_client=DryRunKaliClient(), artifact_store=artifact_store)

    checkpointer = await build_async_configured_checkpointer(args.checkpoint_backend)
    thread_id = args.thread_id or args.project
    graph = build_assessment_graph(
        planner=planner,
        recon=recon,
        discovery_reasoner=discovery_reasoner,
        checkpointer=checkpointer,
    )
    config = checkpoint_config(thread_id) if checkpointer is not None else None
    result = await graph.ainvoke(
        {
            "project_id": args.project,
            "target_scope": args.target,
            "rules_of_engagement": {
                "authorized": True,
                "allow_exploit": False,
                "allow_bruteforce": False,
                "mode": "dry-run" if args.dry_run else "docker",
            },
        },
        config=config,
    )

    report = result.get("report_markdown", "")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Report saved to {output_path}")

    print(report)
    return result


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
