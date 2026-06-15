from __future__ import annotations

from pathlib import Path


DEFAULT_ARTIFACT_ROOT = Path("data") / "artifacts"


def flow_artifact_dir(flow_id: str, root: Path = DEFAULT_ARTIFACT_ROOT) -> Path:
    return root / flow_id


def action_artifact_dir(flow_id: str, action_id: str, root: Path = DEFAULT_ARTIFACT_ROOT) -> Path:
    return flow_artifact_dir(flow_id, root) / "actions" / action_id

