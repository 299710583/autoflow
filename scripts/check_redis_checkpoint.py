from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.graph import END, StateGraph

from autoflow.graph.checkpoints import build_async_redis_checkpointer, checkpoint_config
from autoflow.settings import settings


class SmokeState(TypedDict, total=False):
    value: int


async def smoke_node(state: SmokeState) -> SmokeState:
    return {"value": state.get("value", 0) + 1}


async def run() -> int:
    parser = argparse.ArgumentParser(description="Check LangGraph Redis checkpoint integration.")
    parser.add_argument("--url", default=settings.redis_url, help="Redis URL.")
    parser.add_argument("--thread-id", default="autoflow-redis-checkpoint-healthcheck")
    args = parser.parse_args()

    saver = await build_async_redis_checkpointer(redis_url=args.url, setup=True)
    graph_builder = StateGraph(SmokeState)
    graph_builder.add_node("smoke", smoke_node)
    graph_builder.set_entry_point("smoke")
    graph_builder.add_edge("smoke", END)
    graph = graph_builder.compile(checkpointer=saver)

    config = checkpoint_config(args.thread_id)
    output = await graph.ainvoke({"value": 1}, config=config)
    checkpoint = await saver.aget_tuple(config)
    if output.get("value") != 2 or checkpoint is None:
        print("Redis checkpoint check failed")
        return 1

    print(f"Redis checkpoint URL: {args.url}")
    print(f"Thread ID: {args.thread_id}")
    print(f"Graph output: {output}")
    print(f"Checkpoint ID: {checkpoint.config['configurable'].get('checkpoint_id')}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
