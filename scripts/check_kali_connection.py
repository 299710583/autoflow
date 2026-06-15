"""Check connectivity to the configured Kali execution environment."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoflow.executor.kali_client import KaliClient
from autoflow.settings import settings


def main() -> None:
    print(f"Checking Kali SSH connection: {settings.kali_username}@{settings.kali_host}:{settings.kali_port}")

    try:
        result = KaliClient().check_connection()
    except Exception as exc:
        print(f"FAILED: {exc}")
        raise SystemExit(1) from exc

    if result.succeeded:
        print("OK")
        print(result.stdout.strip())
        return

    print(f"FAILED: exit_code={result.exit_code}")
    if result.stderr:
        print(result.stderr.strip())
    raise SystemExit(result.exit_code or 1)


if __name__ == "__main__":
    main()
