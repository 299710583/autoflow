from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from autoflow.api.schemas import ApprovalDecisionRequest, ApprovalRequestRead
from autoflow.policy.approval import ApprovalStatus, approval_store


router = APIRouter(prefix="/approvals", tags=["approvals"])


@router.get("", response_model=list[ApprovalRequestRead])
def list_approvals(status: str | None = Query(default=None)) -> list[dict]:
    parsed_status = None
    if status is not None:
        try:
            parsed_status = ApprovalStatus(status)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Unknown approval status '{status}'") from exc
    return [item.to_dict() for item in approval_store.list(parsed_status)]


@router.post("/{action_id}/approve", response_model=ApprovalRequestRead)
def approve_action(action_id: str, request: ApprovalDecisionRequest) -> dict:
    try:
        item = approval_store.approve(
            action_id,
            decided_by=request.decided_by,
            reason=request.reason,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return item.to_dict()


@router.post("/{action_id}/reject", response_model=ApprovalRequestRead)
def reject_action(action_id: str, request: ApprovalDecisionRequest) -> dict:
    try:
        item = approval_store.reject(
            action_id,
            decided_by=request.decided_by,
            reason=request.reason,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return item.to_dict()
