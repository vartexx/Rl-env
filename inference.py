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
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or HF_TOKEN is required")

    return OpenAI(api_key=api_key, base_url=API_BASE_URL)


def model_name() -> str:
    return MODEL_NAME


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

    log_start(task=task_id, env=BENCHMARK, model=model)

    step_index = 0
    rewards: List[float] = []
    success = False

    try:
        done = False
        while not done:
            step_index += 1
            action = choose_action(client, model, obs.model_dump())
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
