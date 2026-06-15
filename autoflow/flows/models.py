from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FlowStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ActionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


class ArtifactType(str, Enum):
    RAW_OUTPUT = "raw_output"
    STRUCTURED_RESULT = "structured_result"
    REPORT = "report"
    SCREENSHOT = "screenshot"
    LOG = "log"
    OTHER = "other"


class MemoryKind(str, Enum):
    OBSERVATION = "observation"
    FINDING = "finding"
    DECISION = "decision"
    REMEDIATION = "remediation"
    LESSON = "lesson"


class FindingSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FindingStatus(str, Enum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    EXPLOITABLE = "exploitable"
    FALSE_POSITIVE = "false_positive"


class ValidationPlanStatus(str, Enum):
    PLANNED = "planned"
    PENDING_APPROVAL = "pending_approval"
    EXECUTED = "executed"
    COMPLETED = "completed"
    FAILED = "failed"


class ValidationResultStatus(str, Enum):
    VALIDATED = "validated"
    FALSE_POSITIVE = "false_positive"
    INCONCLUSIVE = "inconclusive"


class Finding(BaseModel):
    id: str = Field(default_factory=lambda: new_id("finding"))
    title: str
    status: FindingStatus = FindingStatus.CANDIDATE
    severity: FindingSeverity = FindingSeverity.INFO
    confidence: FindingConfidence = FindingConfidence.MEDIUM
    target: str
    description: str
    evidence: list[str] = Field(default_factory=list)
    recommendation: str = ""
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class AttackSurface(BaseModel):
    id: str = Field(default_factory=lambda: new_id("surface"))
    target: str
    surface_type: str
    technology: str = ""
    entrypoints: list[str] = Field(default_factory=list)
    related_assets: list[str] = Field(default_factory=list)
    related_findings: list[str] = Field(default_factory=list)
    rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class TestPlanAction(BaseModel):
    id: str = Field(default_factory=lambda: new_id("test_action"))
    name: str
    action_kind: str = "tool"
    tool: str
    profile: str
    target: str
    risk_level: RiskLevel = RiskLevel.LOW
    requires_approval: bool = False
    expected_impact: str = ""
    rationale: str = ""
    args: dict[str, str] = Field(default_factory=dict)
    script_template: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TestPlan(BaseModel):
    id: str = Field(default_factory=lambda: new_id("test_plan"))
    target: str
    strategy: str
    angle: str
    risk_level: RiskLevel = RiskLevel.LOW
    requires_approval: bool = False
    related_findings: list[str] = Field(default_factory=list)
    actions: list[TestPlanAction] = Field(default_factory=list)
    rationale: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationPlan(BaseModel):
    id: str = Field(default_factory=lambda: new_id("validation_plan"))
    finding_id: str
    target: str
    objective: str
    risk_level: RiskLevel = RiskLevel.MEDIUM
    requires_approval: bool = True
    status: ValidationPlanStatus = ValidationPlanStatus.PLANNED
    actions: list[TestPlanAction] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    failure_criteria: list[str] = Field(default_factory=list)
    rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ValidationResult(BaseModel):
    id: str = Field(default_factory=lambda: new_id("validation_result"))
    finding_id: str
    validation_plan_id: str
    status: ValidationResultStatus = ValidationResultStatus.INCONCLUSIVE
    confidence: FindingConfidence = FindingConfidence.MEDIUM
    impact: str = ""
    reproduction_steps: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    executed_action_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class Artifact(BaseModel):
    id: str = Field(default_factory=lambda: new_id("artifact"))
    action_id: str | None = None
    type: ArtifactType = ArtifactType.OTHER
    path: str
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class Action(BaseModel):
    id: str = Field(default_factory=lambda: new_id("action"))
    subtask_id: str | None = None
    tool: str
    intent: dict[str, Any] = Field(default_factory=dict)
    status: ActionStatus = ActionStatus.PENDING
    risk_level: RiskLevel = RiskLevel.LOW
    command_preview: str | None = None
    result_summary: str = ""
    error: str | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def mark_started(self) -> None:
        self.status = ActionStatus.RUNNING
        self.started_at = utc_now()

    def mark_succeeded(self, summary: str = "") -> None:
        self.status = ActionStatus.SUCCEEDED
        self.result_summary = summary
        self.finished_at = utc_now()

    def mark_failed(self, error: str) -> None:
        self.status = ActionStatus.FAILED
        self.error = error
        self.finished_at = utc_now()


class SubTask(BaseModel):
    id: str = Field(default_factory=lambda: new_id("subtask"))
    task_id: str | None = None
    agent: str
    objective: str
    status: TaskStatus = TaskStatus.PENDING
    risk_level: RiskLevel = RiskLevel.LOW
    actions: list[Action] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_action(self, action: Action) -> Action:
        action.subtask_id = self.id
        self.actions.append(action)
        self.updated_at = utc_now()
        return action


class AssessmentTask(BaseModel):
    id: str = Field(default_factory=lambda: new_id("task"))
    flow_id: str | None = None
    type: str
    target: str
    objective: str = ""
    status: TaskStatus = TaskStatus.PENDING
    risk_level: RiskLevel = RiskLevel.LOW
    priority: int = 50
    subtasks: list[SubTask] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_subtask(self, subtask: SubTask) -> SubTask:
        subtask.task_id = self.id
        self.subtasks.append(subtask)
        self.updated_at = utc_now()
        return subtask


class MemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: new_id("memory"))
    flow_id: str | None = None
    kind: MemoryKind = MemoryKind.OBSERVATION
    content: str
    source: str = ""
    references: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class AssessmentFlow(BaseModel):
    id: str = Field(default_factory=lambda: new_id("flow"))
    name: str
    target_scope: list[str] = Field(default_factory=list)
    rules_of_engagement: dict[str, Any] = Field(default_factory=dict)
    status: FlowStatus = FlowStatus.CREATED
    tasks: list[AssessmentTask] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    validation_plans: list[ValidationPlan] = Field(default_factory=list)
    validation_results: list[ValidationResult] = Field(default_factory=list)
    attack_surfaces: list[AttackSurface] = Field(default_factory=list)
    test_plans: list[TestPlan] = Field(default_factory=list)
    memories: list[MemoryItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_task(self, task: AssessmentTask) -> AssessmentTask:
        task.flow_id = self.id
        self.tasks.append(task)
        self.updated_at = utc_now()
        return task

    def add_memory(self, memory: MemoryItem) -> MemoryItem:
        memory.flow_id = self.id
        self.memories.append(memory)
        self.updated_at = utc_now()
        return memory

    def add_finding(self, finding: Finding) -> Finding:
        self.findings.append(finding)
        self.updated_at = utc_now()
        return finding

    def add_validation_plan(self, validation_plan: ValidationPlan) -> ValidationPlan:
        self.validation_plans.append(validation_plan)
        self.updated_at = utc_now()
        return validation_plan

    def add_validation_result(self, validation_result: ValidationResult) -> ValidationResult:
        self.validation_results.append(validation_result)
        self.updated_at = utc_now()
        return validation_result

    def add_attack_surface(self, attack_surface: AttackSurface) -> AttackSurface:
        self.attack_surfaces.append(attack_surface)
        self.updated_at = utc_now()
        return attack_surface

    def add_test_plan(self, test_plan: TestPlan) -> TestPlan:
        self.test_plans.append(test_plan)
        self.updated_at = utc_now()
        return test_plan
