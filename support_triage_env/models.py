from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class TicketCategory(str, Enum):
    BILLING = "billing"
    BUG = "bug"
    ACCOUNT = "account"
    SALES = "sales"
    ABUSE = "abuse"


class Ticket(BaseModel):
    ticket_id: str
    subject: str
    message: str
    true_category: TicketCategory
    true_priority: int = Field(ge=1, le=5)
    customer_tier: Literal["free", "pro", "enterprise"]
    sla_hours_left: int = Field(ge=0, le=72)
    status: TicketStatus = TicketStatus.OPEN
    assigned_team: Optional[str] = None
    predicted_category: Optional[TicketCategory] = None
    predicted_priority: Optional[int] = Field(default=None, ge=1, le=5)
    response_template: Optional[str] = None
    resolution_code: Optional[str] = None
    escalation_reason: Optional[str] = None


class ActionType(str, Enum):
    CLASSIFY = "classify"
    ASSIGN = "assign"
    RESPOND = "respond"
    RESOLVE = "resolve"
    ESCALATE = "escalate"
    NOOP = "noop"


class Action(BaseModel):
    action_type: ActionType
    ticket_id: Optional[str] = None
    predicted_category: Optional[TicketCategory] = None
    predicted_priority: Optional[int] = Field(default=None, ge=1, le=5)
    team: Optional[str] = None
    response_template: Optional[str] = None
    resolution_code: Optional[str] = None
    escalation_reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_by_action_type(self) -> "Action":
        if self.action_type == ActionType.NOOP:
            return self

        if self.ticket_id is None:
            raise ValueError("ticket_id is required for non-noop actions")

        if self.action_type == ActionType.CLASSIFY:
            if self.predicted_category is None or self.predicted_priority is None:
                raise ValueError("classify requires predicted_category and predicted_priority")
        elif self.action_type == ActionType.ASSIGN:
            if self.team is None:
                raise ValueError("assign requires team")
        elif self.action_type == ActionType.RESPOND:
            if self.response_template is None:
                raise ValueError("respond requires response_template")
        elif self.action_type == ActionType.RESOLVE:
            if self.resolution_code is None:
                raise ValueError("resolve requires resolution_code")
        elif self.action_type == ActionType.ESCALATE:
            if self.escalation_reason is None:
                raise ValueError("escalate requires escalation_reason")

        return self


class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    reason: str
    components: Dict[str, float] = Field(default_factory=dict)


class ObservationTicket(BaseModel):
    ticket_id: str
    subject: str
    message: str
    customer_tier: Literal["free", "pro", "enterprise"]
    sla_hours_left: int
    status: TicketStatus
    predicted_category: Optional[TicketCategory] = None
    predicted_priority: Optional[int] = None
    assigned_team: Optional[str] = None


class Observation(BaseModel):
    task_id: str
    step_count: int
    max_steps: int
    pending_tickets: int
    resolved_tickets: int
    escalated_tickets: int
    tickets: List[ObservationTicket]
    instruction: str


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class EnvState(BaseModel):
    task_id: str
    task_name: str
    step_count: int
    max_steps: int
    tickets: List[Ticket]
    action_log: List[Action]
    cumulative_reward: float
    done: bool
