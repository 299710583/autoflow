"""Check whether the AutoFlow tool Docker image contains expected tools."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass


DEFAULT_IMAGE = "autoflow-kali-tools:latest"


@dataclass(frozen=True)
class ToolCheck:
    name: str
    command: str


CHECKS = [
    ToolCheck("nmap", "command -v nmap && nmap --version | head -n 1"),
    ToolCheck("whatweb", "command -v whatweb && whatweb --version | head -n 1"),
    ToolCheck("httpx-toolkit", "command -v httpx-toolkit && httpx-toolkit -version 2>&1 | head -n 1"),
    ToolCheck("nikto", "command -v nikto && nikto -Version | head -n 1"),
    ToolCheck("sslscan", "command -v sslscan && sslscan --version | head -n 1"),
    ToolCheck("testssl", "command -v testssl && testssl --version 2>&1 | head -n 1"),
    ToolCheck("nuclei", "command -v nuclei && nuclei -version | head -n 1"),
    ToolCheck("naabu", "command -v naabu && naabu -version 2>&1 | head -n 1"),
    ToolCheck("subfinder", "command -v subfinder && subfinder -version 2>&1 | head -n 1"),
    ToolCheck("wafw00f", "command -v wafw00f && wafw00f --version | head -n 1"),
    ToolCheck("gobuster", "command -v gobuster && gobuster version | head -n 1"),
    ToolCheck("ffuf", "command -v ffuf && ffuf -V | head -n 1"),
    ToolCheck("feroxbuster", "command -v feroxbuster && feroxbuster --version | head -n 1"),
    ToolCheck("dirb", "command -v dirb && dirb 2>&1 | head -n 1"),
    ToolCheck("dirsearch", "command -v dirsearch && dirsearch --version | head -n 1"),
    ToolCheck("sqlmap", "command -v sqlmap && sqlmap --version | head -n 1"),
    ToolCheck("hydra", "command -v hydra && hydra -h 2>&1 | head -n 1"),
    ToolCheck("medusa", "command -v medusa && medusa -h 2>&1 | head -n 1"),
    ToolCheck("smbclient", "command -v smbclient && smbclient --version | head -n 1"),
    ToolCheck("enum4linux", "command -v enum4linux && enum4linux -h 2>&1 | head -n 1"),
    ToolCheck("smbmap", "command -v smbmap && smbmap --version | head -n 1"),
    ToolCheck("trivy", "command -v trivy && trivy --version | head -n 1"),
    ToolCheck("bandit", "command -v bandit && bandit --version 2>&1 | head -n 1"),
    ToolCheck("gitleaks", "command -v gitleaks && gitleaks version 2>&1 | head -n 1"),
    ToolCheck("semgrep", "command -v semgrep && semgrep --version | head -n 1"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check tools installed in the AutoFlow Docker image.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image tag to check.")
    return parser.parse_args()


def run_check(image: str, check: ToolCheck) -> tuple[bool, str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "bash",
        image,
        "-lc",
        check.command,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = (completed.stdout or completed.stderr).strip()
    return completed.returncode == 0, output


def main() -> None:
    args = parse_args()
    failures: list[str] = []

    for check in CHECKS:
        ok, output = run_check(args.image, check)
        status = "OK" if ok else "MISSING"
        print(f"[{status}] {check.name}: {output}")
        if not ok:
            failures.append(check.name)

    if failures:
        joined = ", ".join(failures)
        raise SystemExit(f"Missing tools in image '{args.image}': {joined}")


if __name__ == "__main__":
    main()
