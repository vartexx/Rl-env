from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .models import Ticket, TicketCategory


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    name: str
    difficulty: str
    instruction: str
    tickets: List[Ticket]
    max_steps: int


def get_tasks() -> List[TaskDefinition]:
    return [easy_task(), medium_task(), hard_task()]


def easy_task() -> TaskDefinition:
    return TaskDefinition(
        task_id="triage_easy_password_reset",
        name="Single Password Reset",
        difficulty="easy",
        instruction=(
            "Handle a single account-access ticket end-to-end. "
            "Correctly classify, assign to account_ops, respond, and resolve."
        ),
        tickets=[
            Ticket(
                ticket_id="T-100",
                subject="Locked out after phone change",
                message=(
                    "I enabled 2FA and changed my phone. Now I cannot log in and have "
                    "a demo in 2 hours. Please help urgently."
                ),
                true_category=TicketCategory.ACCOUNT,
                true_priority=4,
                customer_tier="pro",
                sla_hours_left=6,
            )
        ],
        max_steps=6,
    )


def medium_task() -> TaskDefinition:
    return TaskDefinition(
        task_id="triage_medium_mixed_queue",
        name="Mixed Queue with SLA Pressure",
        difficulty="medium",
        instruction=(
            "Handle a mixed queue. Abuse must be escalated to trust_safety. "
            "Billing requires billing_ops and bug requires engineering."
        ),
        tickets=[
            Ticket(
                ticket_id="T-201",
                subject="Repeated harassment in comments",
                message=(
                    "Another user keeps posting threats under my public profile. "
                    "I already reported this twice."
                ),
                true_category=TicketCategory.ABUSE,
                true_priority=5,
                customer_tier="enterprise",
                sla_hours_left=2,
            ),
            Ticket(
                ticket_id="T-202",
                subject="Charged twice this month",
                message=(
                    "My card was charged two times for March. Please issue a refund if this "
                    "is a mistake."
                ),
                true_category=TicketCategory.BILLING,
                true_priority=4,
                customer_tier="pro",
                sla_hours_left=8,
            ),
            Ticket(
                ticket_id="T-203",
                subject="Export button returns 500",
                message=(
                    "CSV export crashes with a 500 error on every report. Happens in Chrome and Edge."
                ),
                true_category=TicketCategory.BUG,
                true_priority=3,
                customer_tier="free",
                sla_hours_left=14,
            ),
        ],
        max_steps=14,
    )


def hard_task() -> TaskDefinition:
    return TaskDefinition(
        task_id="triage_hard_backlog_optimization",
        name="Backlog Optimization Under Constraints",
        difficulty="hard",
        instruction=(
            "You are handling a high-pressure backlog. Prioritize by impact and SLA. "
            "Abuse must be escalated, critical bugs routed quickly, and high-value customer "
            "tickets should not breach SLA."
        ),
        tickets=[
            Ticket(
                ticket_id="T-301",
                subject="Phishing links in marketplace listing",
                message=(
                    "Several listings are redirecting to phishing pages pretending to be us. "
                    "Users are losing account access."
                ),
                true_category=TicketCategory.ABUSE,
                true_priority=5,
                customer_tier="enterprise",
                sla_hours_left=1,
            ),
            Ticket(
                ticket_id="T-302",
                subject="SAML login outage",
                message=(
                    "All users in our org are blocked from SAML SSO after your 09:10 deploy."
                ),
                true_category=TicketCategory.BUG,
                true_priority=5,
                customer_tier="enterprise",
                sla_hours_left=2,
            ),
            Ticket(
                ticket_id="T-303",
                subject="Need invoice with tax ID",
                message=(
                    "Our finance team needs a corrected invoice with VAT number today."
                ),
                true_category=TicketCategory.BILLING,
                true_priority=3,
                customer_tier="pro",
                sla_hours_left=10,
            ),
            Ticket(
                ticket_id="T-304",
                subject="Can we get annual enterprise pricing",
                message=(
                    "We are evaluating vendors and need pricing plus security documents this week."
                ),
                true_category=TicketCategory.SALES,
                true_priority=2,
                customer_tier="free",
                sla_hours_left=20,
            ),
            Ticket(
                ticket_id="T-305",
                subject="Password reset loop",
                message=(
                    "Reset link says success but login still fails with invalid credentials."
                ),
                true_category=TicketCategory.ACCOUNT,
                true_priority=4,
                customer_tier="pro",
                sla_hours_left=7,
            ),
            Ticket(
                ticket_id="T-306",
                subject="Webhook retries delayed by 30 min",
                message=(
                    "Our incident pipeline is delayed because webhook retries are queued too slowly."
                ),
                true_category=TicketCategory.BUG,
                true_priority=4,
                customer_tier="enterprise",
                sla_hours_left=4,
            ),
        ],
        max_steps=24,
    )
