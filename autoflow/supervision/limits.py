from pydantic import BaseModel


class SupervisionLimits(BaseModel):
    same_tool_limit: int = 3
    total_action_limit: int = 10
    no_progress_limit: int = 3
    require_reflection: bool = True

