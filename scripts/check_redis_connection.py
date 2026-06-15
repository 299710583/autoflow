from __future__ import annotations

import argparse
import sys
from pathlib import Path

from redis import Redis
from redis.exceptions import RedisError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoflow.settings import settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Check AutoFlow Redis connectivity.")
    parser.add_argument("--url", default=settings.redis_url, help="Redis URL, for example redis://192.168.34.191:6379/0")
    parser.add_argument("--prefix", default=settings.redis_key_prefix, help="AutoFlow Redis key prefix.")
    args = parser.parse_args()

    key = f"{args.prefix}:healthcheck"
    try:
        client = Redis.from_url(args.url, decode_responses=True)
        pong = client.ping()
        client.set(key, "ok", ex=60)
        value = client.get(key)
    except RedisError as exc:
        print(f"Redis connection failed: {exc}")
        return 1

    print(f"Redis ping: {pong}")
    print(f"Redis URL: {args.url}")
    print(f"Healthcheck key: {key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
