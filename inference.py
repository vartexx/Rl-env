from __future__ import annotations

import json
import os
from typing import Dict, List

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


def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    base_url = os.getenv("API_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def model_name() -> str:
    value = os.getenv("MODEL_NAME")
    if not value:
        raise RuntimeError("MODEL_NAME is required")
    return value


def choose_action(client: OpenAI, model: str, obs: Dict) -> Action:
    user_prompt = (
        "Current observation:\n"
        f"{json.dumps(obs, indent=2)}\n\n"
        "Pick one next action as JSON only."
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        seed=7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content
    if not content:
        return Action(action_type="noop")

    payload = json.loads(content)
    return Action.model_validate(payload)


def run_task(client: OpenAI, model: str, task_id: str) -> Dict[str, float]:
    env = SupportTriageEnv(task_id=task_id)
    obs = env.reset(task_id=task_id)

    done = False
    while not done:
        action = choose_action(client, model, obs.model_dump())
        result = env.step(action)
        obs = result.observation
        done = result.done

    return grade_task(env.state())


def main() -> None:
    client = build_client()
    model = model_name()

    env = SupportTriageEnv()
    task_ids: List[str] = env.task_ids()

    scores: Dict[str, Dict[str, float]] = {}
    for task_id in task_ids:
        scores[task_id] = run_task(client, model, task_id)

    macro = sum(item["score"] for item in scores.values()) / max(1, len(scores))
    output = {
        "model": model,
        "tasks": scores,
        "macro_average": round(macro, 4),
    }

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
