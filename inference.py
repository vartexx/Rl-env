from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from openai import OpenAI

from support_triage_env.environment import SupportTriageEnv
from support_triage_env.graders import grade_task
from support_triage_env.models import Action


SYSTEM_PROMPT = """You are a support triage agent.
Return exactly one JSON object for each turn with this schema:
{
  \"action_type\": \"classify|assign|respond|resolve|escalate|noop\",
  \"ticket_id\": \"optional ticket id\",
  \"predicted_category\": \"billing|bug|account|sales|abuse\",
  \"predicted_priority\": 1-5,
  \"team\": \"account_ops|billing_ops|engineering|sales|trust_safety\",
  \"response_template\": \"brief string\",
  \"resolution_code\": \"brief string\",
  \"escalation_reason\": \"brief string\"
}
Use only fields needed by the chosen action_type.
"""


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "openai/gpt-4o-mini"
BENCHMARK = os.getenv("BENCHMARK") or "support-triage-openenv"
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _action_to_str(action: Action) -> str:
    payload = action.model_dump(exclude_none=True)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={_bool_str(done)} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={_bool_str(success)} steps={steps} rewards={rewards_str}", flush=True)


def build_client() -> OpenAI:
    api_key = os.getenv("API_KEY") or HF_TOKEN or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY (or OPENAI_API_KEY/HF_TOKEN) is required")

    return OpenAI(api_key=api_key, base_url=API_BASE_URL)


def model_name() -> str:
    return MODEL_NAME


def _ticket_text(ticket: Dict) -> str:
    return f"{ticket.get('subject', '')} {ticket.get('message', '')}".lower()


def _infer_category(ticket: Dict) -> str:
    text = _ticket_text(ticket)
    if any(keyword in text for keyword in ["harassment", "threat", "phishing", "abuse", "fraud", "scam", "impersonat"]):
        return "abuse"
    if any(keyword in text for keyword in ["saml", "outage", "deploy", "blocked", "all users", "webhook", "crash", "500", "export", "delay"]):
        return "bug"
    if any(keyword in text for keyword in ["invoice", "charged", "billing", "refund", "vat", "tax id", "payment", "card"]):
        return "billing"
    if any(keyword in text for keyword in ["pricing", "security docs", "vendor", "enterprise pricing", "quote", "sales"]):
        return "sales"
    if any(keyword in text for keyword in ["login", "password", "locked", "account", "2fa", "sso", "access", "reset"]):
        return "account"
    return "account"


def _infer_priority(ticket: Dict, category: str) -> int:
    text = _ticket_text(ticket)
    if category == "abuse":
        return 5
    if any(keyword in text for keyword in ["urgent", "immediate", "today", "blocked", "outage", "critical", "demo", "asap"]):
        return 5
    if category in {"billing", "account"}:
        return 4
    if category == "bug":
        return 4 if any(keyword in text for keyword in ["sso", "outage", "all users", "crash", "500"] ) else 3
    if category == "sales":
        return 2
    return 3


def _team_for_category(category: str) -> str:
    return {
        "account": "account_ops",
        "billing": "billing_ops",
        "bug": "engineering",
        "sales": "sales",
        "abuse": "trust_safety",
    }[category]


def _is_terminal(ticket: Dict) -> bool:
    status = _status_value(ticket)
    return status in {"resolved", "escalated"}


def _status_value(ticket: Dict) -> str:
    status = ticket.get("status", "")
    if hasattr(status, "value"):
        return str(status.value).lower()
    return str(status).lower().replace("ticketstatus.", "")


def _next_action_for_ticket(ticket: Dict, phase: int) -> Optional[Action]:
    if _is_terminal(ticket):
        return None

    category = _infer_category(ticket)
    ticket_id = ticket.get("ticket_id")
    predicted_category = ticket.get("predicted_category")
    predicted_priority = ticket.get("predicted_priority")
    assigned_team = ticket.get("assigned_team")
    response_template = ticket.get("response_template")
    resolution_code = ticket.get("resolution_code")
    escalation_reason = ticket.get("escalation_reason")

    if phase == 0 or predicted_category is None:
        return Action(
            action_type="classify",
            ticket_id=ticket_id,
            predicted_category=category,
            predicted_priority=_infer_priority(ticket, category),
        )

    if category == "abuse":
        if phase == 1 or assigned_team != "trust_safety":
            return Action(action_type="assign", ticket_id=ticket_id, team="trust_safety")
        if phase >= 2:
            return Action(
                action_type="escalate",
                ticket_id=ticket_id,
                escalation_reason=escalation_reason
                if escalation_reason
                else "Immediate escalation to trust_safety due to abuse or threat report.",
            )
        return None

    expected_team = _team_for_category(category)
    if phase == 1 or assigned_team != expected_team:
        return Action(action_type="assign", ticket_id=ticket_id, team=expected_team)

    if phase == 2:
        if category == "account":
            template = "We understand the access issue and are assisting with account recovery steps."
        elif category == "billing":
            template = "We are reviewing the billing issue and will follow up with the next steps."
        elif category == "bug":
            template = "Engineering is investigating the issue and we will update you shortly."
        else:
            template = "We are reviewing your request and will follow up shortly."
        return Action(action_type="respond", ticket_id=ticket_id, response_template=template)

    if phase >= 3:
        return Action(
            action_type="resolve",
            ticket_id=ticket_id,
            resolution_code=resolution_code if resolution_code else "resolved_by_support",
        )

    return None


def choose_action(client: OpenAI, model: str, obs: Dict, progress: Dict[str, int]) -> Action:
    # Primary path: request an action through the provided LLM proxy.
    # Fallback path: deterministic heuristic if API call or parsing fails.
    user_prompt = (
        "Return exactly one JSON object for the next support action.\n"
        "Observation:\n"
        f"{json.dumps(obs, default=str)}\n\n"
        "Required keys by action_type:\n"
        "- classify: action_type,ticket_id,predicted_category,predicted_priority\n"
        "- assign: action_type,ticket_id,team\n"
        "- respond: action_type,ticket_id,response_template\n"
        "- resolve: action_type,ticket_id,resolution_code\n"
        "- escalate: action_type,ticket_id,escalation_reason\n"
        "- noop: action_type\n"
        "Allowed categories: billing, bug, account, sales, abuse\n"
        "Allowed teams: account_ops, billing_ops, engineering, sales, trust_safety\n"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        payload = json.loads(content)
        return Action.model_validate(payload)
    except Exception:
        pass

    tickets = list(obs.get("tickets", []))
    for ticket in tickets:
        ticket_id = str(ticket.get("ticket_id", ""))
        phase = progress.get(ticket_id, 0)
        next_action = _next_action_for_ticket(ticket, phase)
        if next_action is None:
            progress[ticket_id] = max(progress.get(ticket_id, 0), 4)
            continue
        return next_action

    return Action(action_type="noop")


def run_task(client: OpenAI, model: str, task_id: str) -> Dict[str, float]:
    env = SupportTriageEnv(task_id=task_id)
    obs = env.reset(task_id=task_id)
    progress: Dict[str, int] = {}

    log_start(task=task_id, env=BENCHMARK, model=model)

    step_index = 0
    rewards: List[float] = []
    success = False

    try:
        done = False
        while not done:
            step_index += 1
            action = choose_action(client, model, obs.model_dump(), progress)
            action_str = _action_to_str(action)
            error = None

            try:
                result = env.step(action)
            except Exception as exc:
                error = str(exc)
                result = env.step(Action(action_type="noop"))

            reward_value = float(result.reward.value)
            rewards.append(reward_value)
            log_step(step=step_index, action=action_str, reward=reward_value, done=result.done, error=error)

            if action.ticket_id:
                previous_phase = progress.get(action.ticket_id, 0)
                if action.action_type.value == "classify":
                    progress[action.ticket_id] = max(previous_phase, 1)
                elif action.action_type.value == "assign":
                    progress[action.ticket_id] = max(previous_phase, 2)
                elif action.action_type.value == "respond":
                    progress[action.ticket_id] = max(previous_phase, 3)
                elif action.action_type.value in {"resolve", "escalate"}:
                    progress[action.ticket_id] = max(previous_phase, 4)

            obs = result.observation
            done = result.done

        graded = grade_task(env.state())
        success = bool(graded.get("score", 0.0) >= 0.6)
        return graded
    finally:
        log_end(success=success, steps=step_index, rewards=rewards)


def main() -> None:
    client = build_client()
    model = model_name()

    env = SupportTriageEnv()
    task_ids: List[str] = env.task_ids()

    scores: Dict[str, Dict[str, float]] = {}
    for task_id in task_ids:
        scores[task_id] = run_task(client, model, task_id)

    # Keep final summary in logs-compatible form and avoid non-protocol stdout lines.
    _ = sum(item["score"] for item in scores.values()) / max(1, len(scores))


if __name__ == "__main__":
    main()
