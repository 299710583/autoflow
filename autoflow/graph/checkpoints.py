from __future__ import annotations

from typing import Any

from autoflow.settings import settings


def build_memory_checkpointer():
    """Build the built-in in-process LangGraph checkpointer."""

    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


def build_redis_checkpointer(
    redis_url: str | None = None,
    *,
    setup: bool = True,
    ttl_seconds: int | None = None,
    checkpoint_prefix: str | None = None,
    checkpoint_write_prefix: str | None = None,
    **kwargs: Any,
):
    """Build the Redis LangGraph checkpointer."""

    try:
        from langgraph.checkpoint.redis import RedisSaver  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Redis LangGraph checkpointer is not installed. Install the optional "
            "`langgraph-checkpoint-redis` package, or use build_memory_checkpointer() "
            "while RedisMemoryStore handles runtime memory indexes."
        ) from exc

    ttl = _ttl_config(ttl_seconds)
    saver = RedisSaver(
        redis_url=redis_url or settings.redis_url,
        ttl=ttl,
        checkpoint_prefix=checkpoint_prefix or f"{settings.redis_key_prefix}:checkpoint",
        checkpoint_write_prefix=checkpoint_write_prefix or f"{settings.redis_key_prefix}:checkpoint_write",
        **kwargs,
    )
    if setup:
        saver.setup()
    return saver


async def build_async_redis_checkpointer(
    redis_url: str | None = None,
    *,
    setup: bool = True,
    ttl_seconds: int | None = None,
    checkpoint_prefix: str | None = None,
    checkpoint_write_prefix: str | None = None,
    **kwargs: Any,
):
    """Build the async Redis LangGraph checkpointer for graph.ainvoke/astream."""

    try:
        from langgraph.checkpoint.redis import AsyncRedisSaver  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Redis LangGraph checkpointer is not installed. Install the optional "
            "`langgraph-checkpoint-redis` package, or use build_memory_checkpointer() "
            "while RedisMemoryStore handles runtime memory indexes."
        ) from exc

    ttl = _ttl_config(ttl_seconds)
    saver = AsyncRedisSaver(
        redis_url=redis_url or settings.redis_url,
        ttl=ttl,
        checkpoint_prefix=checkpoint_prefix or f"{settings.redis_key_prefix}:checkpoint",
        checkpoint_write_prefix=checkpoint_write_prefix or f"{settings.redis_key_prefix}:checkpoint_write",
        **kwargs,
    )
    if setup:
        await saver.setup()
    return saver


def build_configured_checkpointer(backend: str | None = None):
    """Build a checkpointer from settings.

    `auto` means Redis when REDIS_ENABLED=true, otherwise no graph checkpointer.
    """

    selected = (backend or settings.checkpoint_backend or "auto").lower()
    if selected == "auto":
        selected = "redis" if settings.redis_enabled else "none"
    if selected in {"", "none", "off", "false"}:
        return None
    if selected == "memory":
        return build_memory_checkpointer()
    if selected == "redis":
        return build_redis_checkpointer()
    raise ValueError(f"Unsupported checkpoint backend: {selected}")


async def build_async_configured_checkpointer(backend: str | None = None):
    """Build an async-compatible checkpointer from settings."""

    selected = (backend or settings.checkpoint_backend or "auto").lower()
    if selected == "auto":
        selected = "redis" if settings.redis_enabled else "none"
    if selected in {"", "none", "off", "false"}:
        return None
    if selected == "memory":
        return build_memory_checkpointer()
    if selected == "redis":
        return await build_async_redis_checkpointer()
    raise ValueError(f"Unsupported checkpoint backend: {selected}")


def checkpoint_config(thread_id: str, checkpoint_ns: str = "") -> dict[str, dict[str, str]]:
    if not thread_id:
        raise ValueError("thread_id is required when using a graph checkpointer")
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}


def _ttl_config(ttl_seconds: int | None) -> dict[str, Any] | None:
    seconds = settings.checkpoint_ttl_seconds if ttl_seconds is None else ttl_seconds
    if seconds <= 0:
        return None
    minutes = max(1, int(seconds / 60))
    return {"default_ttl": minutes, "refresh_on_read": True}
