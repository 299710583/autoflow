from __future__ import annotations

from pathlib import Path

from autoflow.artifacts.paths import DEFAULT_ARTIFACT_ROOT, action_artifact_dir
from autoflow.flows.models import Artifact, ArtifactType


class ArtifactStore:
    def __init__(self, root: Path = DEFAULT_ARTIFACT_ROOT) -> None:
        self.root = root

    def reserve_action_path(self, flow_id: str, action_id: str, filename: str) -> Path:
        directory = action_artifact_dir(flow_id, action_id, self.root)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / filename

    def register(
        self,
        path: Path,
        artifact_type: ArtifactType,
        action_id: str | None = None,
        summary: str = "",
    ) -> Artifact:
        return Artifact(
            action_id=action_id,
            type=artifact_type,
            path=str(path),
            summary=summary,
        )

