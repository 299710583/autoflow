"""Build the local AutoFlow pentest tools Docker image."""

from __future__ import annotations

import subprocess
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE_DIR = ROOT / "docker" / "autoflow-kali-tools"
IMAGE_NAME = "autoflow-kali-tools:latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the AutoFlow pentest tools Docker image.")
    parser.add_argument("--tag", default=IMAGE_NAME, help="Docker image tag to build.")
    parser.add_argument(
        "--base-image",
        default="kalilinux/kali-rolling",
        help="Base image. Use a local Ubuntu image as a fallback when Docker Hub is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        "docker",
        "build",
        "--build-arg",
        f"BASE_IMAGE={args.base_image}",
        "-t",
        args.tag,
        str(DOCKERFILE_DIR),
    ]
    print(" ".join(command))
    completed = subprocess.run(command, cwd=ROOT, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
