from __future__ import annotations

from typing import Dict, Tuple

from .models import EnvState, TicketCategory, TicketStatus


def _ticket_score_components(ticket) -> Dict[str, float]:
    score = {
        "classification": 0.0,
        "assignment": 0.0,
        "response": 0.0,
        "outcome": 0.0,
    }

    if ticket.predicted_category == ticket.true_category:
        score["classification"] = 0.25

    expected_team = {
        TicketCategory.ACCOUNT: "account_ops",
        TicketCategory.BILLING: "billing_ops",
        TicketCategory.BUG: "engineering",
        TicketCategory.SALES: "sales",
        TicketCategory.ABUSE: "trust_safety",
    }[ticket.true_category]

    if ticket.assigned_team == expected_team:
        score["assignment"] = 0.20

    if ticket.response_template is not None:
        score["response"] = 0.15

    if ticket.true_category == TicketCategory.ABUSE:
        if ticket.status == TicketStatus.ESCALATED and ticket.assigned_team == "trust_safety":
            score["outcome"] = 0.40
    else:
        if ticket.status == TicketStatus.RESOLVED:
            score["outcome"] = 0.40

    return score


def grade_state(state: EnvState) -> Tuple[float, Dict[str, float]]:
    if not state.tickets:
        # Return epsilon instead of 0.0 to satisfy validator requirement (0 < score < 1)
        return 0.001, {"empty": 0.001}

    per_ticket_scores = []
    aggregate = {
        "classification": 0.0,
        "assignment": 0.0,
        "response": 0.0,
        "outcome": 0.0,
        "efficiency": 0.0,
    }

    for ticket in state.tickets:
        comp = _ticket_score_components(ticket)
        ticket_score = sum(comp.values())
        per_ticket_scores.append(ticket_score)
        for key, value in comp.items():
            aggregate[key] += value

    n = float(len(state.tickets))
    for key in ["classification", "assignment", "response", "outcome"]:
        aggregate[key] = aggregate[key] / n

    # Efficiency bonus rewards solving tasks in fewer steps than the budget.
    efficiency = max(0.0, 1.0 - (state.step_count / max(1, state.max_steps)))
    aggregate["efficiency"] = 0.10 * efficiency

    score = sum(aggregate.values())
    score = max(0.0, min(1.0, score))
    
    # Ensure score is strictly between 0 and 1 (exclusive)
    epsilon = 0.001
    if score == 0.0:
        score = epsilon
    elif score == 1.0:
        score = 1.0 - epsilon
    
    return score, aggregate


def grade_task(state: EnvState) -> Dict[str, float]:
    score, components = grade_state(state)
    result = {"score": score}
    result.update(components)
    return result
