from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from .graders import grade_task
from .models import (
    Action,
    ActionType,
    EnvState,
    Observation,
    ObservationTicket,
    Reward,
    StepResult,
    Ticket,
    TicketCategory,
    TicketStatus,
)
from .tasks import TaskDefinition, get_tasks


TEAM_BY_CATEGORY = {
    TicketCategory.ACCOUNT: "account_ops",
    TicketCategory.BILLING: "billing_ops",
    TicketCategory.BUG: "engineering",
    TicketCategory.SALES: "sales",
    TicketCategory.ABUSE: "trust_safety",
}


class SupportTriageEnv:
    def __init__(self, task_id: Optional[str] = None):
        self._tasks = {task.task_id: task for task in get_tasks()}
        self._task_order = list(self._tasks.keys())
        self._task_index = 0
        self._task: Optional[TaskDefinition] = None
        self._state: Optional[EnvState] = None

        if task_id is not None and task_id not in self._tasks:
            raise ValueError(f"Unknown task_id: {task_id}")

        self._fixed_task_id = task_id
        self.reset(task_id=task_id)

    def task_ids(self) -> List[str]:
        return list(self._task_order)

    def reset(self, task_id: Optional[str] = None) -> Observation:
        if task_id is None:
            if self._fixed_task_id is not None:
                task_id = self._fixed_task_id
            else:
                task_id = self._task_order[self._task_index]
                self._task_index = (self._task_index + 1) % len(self._task_order)

        task = self._tasks[task_id]
        self._task = task
        self._state = EnvState(
            task_id=task.task_id,
            task_name=task.name,
            step_count=0,
            max_steps=task.max_steps,
            tickets=copy.deepcopy(task.tickets),
            action_log=[],
            cumulative_reward=0.0,
            done=False,
        )
        return self._build_observation()

    def state(self) -> EnvState:
        if self._state is None:
            raise RuntimeError("Environment is not initialized")
        return copy.deepcopy(self._state)

    def step(self, action: Action) -> StepResult:
        if self._state is None or self._task is None:
            raise RuntimeError("Environment is not initialized")
        if self._state.done:
            reward = Reward(value=0.0, reason="episode_done", components={"terminal": 0.0})
            return StepResult(observation=self._build_observation(), reward=reward, done=True, info={"warning": "Episode already done"})

        reward_components: Dict[str, float] = {"progress": 0.0, "penalty": 0.0}
        reward_reason = "action_processed"

        self._state.step_count += 1
        self._state.action_log.append(action)

        if action.action_type == ActionType.NOOP:
            reward_components["penalty"] -= 0.05
            reward_reason = "noop_penalty"
        else:
            ticket = self._find_ticket(action.ticket_id)
            if ticket is None:
                reward_components["penalty"] -= 0.2
                reward_reason = "invalid_ticket"
            else:
                self._apply_action(ticket, action, reward_components)

        if self._state.step_count >= self._state.max_steps:
            self._state.done = True

        if self._all_terminal_tickets():
            self._state.done = True
            reward_components["progress"] += 0.15
            reward_reason = "all_tickets_terminal"

        step_reward = reward_components["progress"] + reward_components["penalty"]
        step_reward = max(-1.0, min(1.0, step_reward))
        self._state.cumulative_reward += step_reward

        info: Dict[str, Any] = {}
        if self._state.done:
            info["grader"] = grade_task(self._state)

        reward = Reward(value=step_reward, reason=reward_reason, components=reward_components)
        return StepResult(
            observation=self._build_observation(),
            reward=reward,
            done=self._state.done,
            info=info,
        )

    def _find_ticket(self, ticket_id: Optional[str]) -> Optional[Ticket]:
        if ticket_id is None or self._state is None:
            return None
        for ticket in self._state.tickets:
            if ticket.ticket_id == ticket_id:
                return ticket
        return None

    def _all_terminal_tickets(self) -> bool:
        assert self._state is not None
        for ticket in self._state.tickets:
            if ticket.true_category == TicketCategory.ABUSE:
                if ticket.status != TicketStatus.ESCALATED:
                    return False
            else:
                if ticket.status != TicketStatus.RESOLVED:
                    return False
        return True

    def _apply_action(self, ticket: Ticket, action: Action, reward_components: Dict[str, float]) -> None:
        if ticket.status in [TicketStatus.RESOLVED, TicketStatus.ESCALATED]:
            reward_components["penalty"] -= 0.08
            return

        if action.action_type == ActionType.CLASSIFY:
            assert action.predicted_category is not None
            assert action.predicted_priority is not None
            ticket.predicted_category = action.predicted_category
            ticket.predicted_priority = action.predicted_priority
            if action.predicted_category == ticket.true_category:
                reward_components["progress"] += 0.10
            else:
                reward_components["penalty"] -= 0.06

            priority_error = abs(action.predicted_priority - ticket.true_priority)
            reward_components["progress"] += max(0.0, 0.06 - (0.02 * priority_error))

        elif action.action_type == ActionType.ASSIGN:
            assert action.team is not None
            ticket.assigned_team = action.team
            expected = TEAM_BY_CATEGORY[ticket.true_category]
            if action.team == expected:
                reward_components["progress"] += 0.10
            else:
                reward_components["penalty"] -= 0.05

        elif action.action_type == ActionType.RESPOND:
            assert action.response_template is not None
            ticket.response_template = action.response_template
            reward_components["progress"] += 0.05

        elif action.action_type == ActionType.RESOLVE:
            assert action.resolution_code is not None
            ticket.resolution_code = action.resolution_code
            if ticket.true_category == TicketCategory.ABUSE:
                reward_components["penalty"] -= 0.25
            else:
                ticket.status = TicketStatus.RESOLVED
                reward_components["progress"] += 0.18

        elif action.action_type == ActionType.ESCALATE:
            assert action.escalation_reason is not None
            ticket.escalation_reason = action.escalation_reason
            if ticket.true_category == TicketCategory.ABUSE:
                ticket.status = TicketStatus.ESCALATED
                ticket.assigned_team = "trust_safety"
                reward_components["progress"] += 0.20
            else:
                reward_components["penalty"] -= 0.06

        if ticket.sla_hours_left > 0:
            ticket.sla_hours_left -= 1
        if ticket.sla_hours_left == 0 and ticket.status in [TicketStatus.OPEN, TicketStatus.IN_PROGRESS]:
            reward_components["penalty"] -= 0.08

        if ticket.status == TicketStatus.OPEN:
            ticket.status = TicketStatus.IN_PROGRESS

    def _build_observation(self) -> Observation:
        assert self._state is not None and self._task is not None
        tickets = [
            ObservationTicket(
                ticket_id=t.ticket_id,
                subject=t.subject,
                message=t.message,
                customer_tier=t.customer_tier,
                sla_hours_left=t.sla_hours_left,
                status=t.status,
                predicted_category=t.predicted_category,
                predicted_priority=t.predicted_priority,
                assigned_team=t.assigned_team,
            )
            for t in self._state.tickets
        ]

        resolved = sum(1 for t in self._state.tickets if t.status == TicketStatus.RESOLVED)
        escalated = sum(1 for t in self._state.tickets if t.status == TicketStatus.ESCALATED)
        pending = len(self._state.tickets) - resolved - escalated

        return Observation(
            task_id=self._task.task_id,
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            pending_tickets=pending,
            resolved_tickets=resolved,
            escalated_tickets=escalated,
            tickets=tickets,
            instruction=self._task.instruction,
        )
